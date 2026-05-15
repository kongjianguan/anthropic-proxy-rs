use crate::config::Config;
use crate::error::{ProxyError, ProxyResult};
use crate::metrics;
use crate::models::{anthropic, openai};
use crate::translate::{pipeline, stream};
use axum::{
    body::Body,
    http::{HeaderMap, HeaderValue},
    response::{IntoResponse, Response},
    Extension, Json,
};
use bytes::Bytes;
use futures::stream::{Stream, StreamExt};
use reqwest::Client;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub async fn proxy_handler(
    Extension(config): Extension<Arc<Config>>,
    Extension(client): Extension<Client>,
    Json(req): Json<anthropic::AnthropicRequest>,
) -> ProxyResult<Response> {
    let is_streaming = req.stream.unwrap_or(false);
    let start = Instant::now();

    tracing::debug!("Received request for model: {}", req.model);
    tracing::debug!("Streaming: {}", is_streaming);
    metrics::request_started(is_streaming);

    if config.verbose {
        tracing::trace!(
            "Incoming Anthropic request: {}",
            serde_json::to_string_pretty(&req).unwrap_or_default()
        );
    }

    let policy = translation_policy(&config);
    let openai_req = pipeline::translate_request(req, &policy)?;

    if config.verbose {
        tracing::trace!(
            "Transformed OpenAI request: {}",
            serde_json::to_string_pretty(&openai_req).unwrap_or_default()
        );
    }

    let result = if is_streaming {
        handle_streaming(config, client, openai_req).await
    } else {
        handle_non_streaming(config, client, openai_req).await
    };

    let status = match &result {
        Ok(resp) => resp.status().as_u16(),
        Err(_) => 500,
    };
    metrics::request_finished(start, status, is_streaming);

    result
}

pub async fn list_models_handler(
    Extension(config): Extension<Arc<Config>>,
    Extension(client): Extension<Client>,
) -> ProxyResult<Response> {
    let urls = config.models_urls();
    let mut last_err = None;

    for url in &urls {
        tracing::debug!("Fetching models from {}", url);

        let mut req_builder = client.get(url).timeout(Duration::from_secs(60));
        if let Some(api_key) = &config.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        match req_builder.send().await {
            Ok(response) if response.status().is_success() => {
                let openai_resp: openai::ModelsListResponse = response.json().await?;
                let anthropic_resp = pipeline::translate_models_list(openai_resp);
                return Ok(Json(anthropic_resp).into_response());
            }
            Ok(response) => {
                let status = response.status();
                let error_text = response
                    .text()
                    .await
                    .unwrap_or_else(|_| "Unknown error".to_string());
                tracing::warn!("Upstream {} returned {}: {}", url, status, error_text);
                if is_retriable_status(status.as_u16()) {
                    last_err = Some(format!("Upstream returned {}: {}", status, error_text));
                    continue;
                }
                return Err(ProxyError::Upstream(format!(
                    "Upstream returned {}: {}",
                    status, error_text
                )));
            }
            Err(err) => {
                tracing::warn!("Failed to reach {}: {:?}", url, err);
                last_err = Some(format!("HTTP error: {}", err));
                continue;
            }
        }
    }

    Err(ProxyError::Upstream(
        last_err.unwrap_or_else(|| "All upstreams failed".to_string()),
    ))
}

fn translation_policy(config: &Config) -> pipeline::TranslationPolicy {
    pipeline::TranslationPolicy {
        reasoning_model: config.reasoning_model.clone(),
        completion_model: config.completion_model.clone(),
        model_map: config.model_map.clone(),
        ignore_terms: config.system_prompt_ignore_terms.clone(),
    }
}

fn is_retriable_status(status: u16) -> bool {
    matches!(status, 429 | 500..=599)
}

async fn handle_non_streaming(
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
) -> ProxyResult<Response> {
    let urls = config.chat_completions_urls();
    let mut last_err = None;

    for url in &urls {
        tracing::debug!(
            "Sending non-streaming request to {} (model: {})",
            url,
            openai_req.model
        );

        let mut req_builder = client
            .post(url)
            .json(&openai_req)
            .timeout(Duration::from_secs(300));

        if let Some(api_key) = &config.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let upstream_start = Instant::now();
        let response = match req_builder.send().await {
            Ok(resp) => {
                metrics::upstream_latency(
                    upstream_start.elapsed().as_secs_f64(),
                    "chat_completions",
                );
                resp
            }
            Err(err) => {
                tracing::warn!("Failed to reach {}: {:?}", url, err);
                metrics::upstream_error("chat_completions");
                last_err = Some(ProxyError::Http(err));
                continue;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            tracing::warn!("Upstream {} returned {}: {}", url, status, error_text);
            metrics::upstream_error("chat_completions");

            if is_retriable_status(status.as_u16()) {
                last_err = Some(ProxyError::Upstream(format!(
                    "Upstream returned {}: {}",
                    status, error_text
                )));
                continue;
            }
            return Err(ProxyError::Upstream(format!(
                "Upstream returned {}: {}",
                status, error_text
            )));
        }

        let openai_resp: openai::OpenAIResponse = response.json().await?;

        metrics::tokens(
            openai_resp.usage.prompt_tokens,
            openai_resp.usage.completion_tokens,
            &openai_req.model,
        );

        if config.verbose {
            tracing::trace!(
                "Received OpenAI response: {}",
                serde_json::to_string_pretty(&openai_resp).unwrap_or_default()
            );
        }

        let anthropic_resp = pipeline::translate_response(openai_resp, &openai_req.model)?;

        if config.verbose {
            tracing::trace!(
                "Transformed Anthropic response: {}",
                serde_json::to_string_pretty(&anthropic_resp).unwrap_or_default()
            );
        }

        return Ok(Json(anthropic_resp).into_response());
    }

    Err(last_err.unwrap_or_else(|| ProxyError::Upstream("All upstreams failed".to_string())))
}

async fn handle_streaming(
    config: Arc<Config>,
    client: Client,
    openai_req: openai::OpenAIRequest,
) -> ProxyResult<Response> {
    let urls = config.chat_completions_urls();
    let mut last_err = None;

    for url in &urls {
        tracing::debug!(
            "Sending streaming request to {} (model: {})",
            url,
            openai_req.model
        );

        let mut req_builder = client
            .post(url)
            .json(&openai_req)
            .timeout(Duration::from_secs(300));

        if let Some(api_key) = &config.api_key {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        }

        let upstream_start = Instant::now();
        let response = match req_builder.send().await {
            Ok(resp) => {
                metrics::upstream_latency(
                    upstream_start.elapsed().as_secs_f64(),
                    "chat_completions",
                );
                resp
            }
            Err(err) => {
                tracing::warn!("Failed to reach {}: {:?}", url, err);
                metrics::upstream_error("chat_completions");
                last_err = Some(ProxyError::Http(err));
                continue;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            tracing::warn!("Upstream {} returned {}: {}", url, status, error_text);
            metrics::upstream_error("chat_completions");

            if is_retriable_status(status.as_u16()) {
                last_err = Some(ProxyError::Upstream(format!(
                    "Upstream returned {}: {}",
                    status, error_text
                )));
                continue;
            }
            return Err(ProxyError::Upstream(format!(
                "Upstream returned {}: {}",
                status, error_text
            )));
        }

        let upstream = response.bytes_stream();
        let sse_stream = create_sse_stream(upstream, openai_req.model.clone());

        let mut headers = HeaderMap::new();
        headers.insert(
            "Content-Type",
            HeaderValue::from_static("text/event-stream"),
        );
        headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
        headers.insert("Connection", HeaderValue::from_static("keep-alive"));

        return Ok((headers, Body::from_stream(sse_stream)).into_response());
    }

    Err(last_err.unwrap_or_else(|| ProxyError::Upstream("All upstreams failed".to_string())))
}

fn serialize_event(event: &anthropic::StreamEvent) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        event.event_type(),
        serde_json::to_string(event).unwrap_or_default()
    )
}

fn create_sse_stream(
    upstream: impl Stream<Item = Result<Bytes, impl std::fmt::Display + Send + 'static>>
        + Send
        + 'static,
    fallback_model: String,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut state = stream::initial_state(fallback_model);

        tokio::pin!(upstream);

        while let Some(chunk) = upstream.next().await {
            match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    buffer.push_str(&text);

                    while let Some(pos) = buffer.find("\n\n") {
                        let line = buffer[..pos].to_string();
                        buffer = buffer[pos + 2..].to_string();

                        if line.trim().is_empty() {
                            continue;
                        }

                        for l in line.lines() {
                            if let Some(data) = l.strip_prefix("data: ") {
                                if data.trim() == "[DONE]" {
                                    for event in stream::translate_done(&mut state) {
                                        yield Ok(Bytes::from(serialize_event(&event)));
                                    }
                                    continue;
                                }

                                if let Ok(chunk) = serde_json::from_str::<openai::StreamChunk>(data) {
                                    for event in stream::translate_chunk(&mut state, &chunk) {
                                        yield Ok(Bytes::from(serialize_event(&event)));
                                    }
                                } else {
                                    tracing::debug!("Ignoring unrecognized upstream stream chunk: {}", data);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Stream error: {}", e);
                    for event in stream::translate_error(format!("Stream error: {}", e)) {
                        yield Ok(Bytes::from(serialize_event(&event)));
                    }
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::create_sse_stream;
    use bytes::Bytes;
    use futures::stream::{self, StreamExt};
    use serde_json::{json, Value};
    use std::fmt;

    #[derive(Debug)]
    struct TestError;
    impl fmt::Display for TestError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test error")
        }
    }

    fn openai_chunk(
        id: &str,
        model: &str,
        content: Option<&str>,
        finish_reason: Option<&str>,
    ) -> String {
        let mut delta = json!({});
        if let Some(c) = content {
            delta["content"] = json!(c);
        }
        let mut choice = json!({ "index": 0, "delta": delta });
        if let Some(fr) = finish_reason {
            choice["finish_reason"] = json!(fr);
        }
        let chunk = json!({
            "id": id,
            "model": model,
            "choices": [choice],
        });
        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())
    }

    fn openai_chunk_with_reasoning(id: &str, model: &str, reasoning: &str) -> String {
        let chunk = json!({
            "id": id,
            "model": model,
            "choices": [{ "index": 0, "delta": { "reasoning": reasoning } }],
        });
        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())
    }

    fn openai_chunk_with_tool_call(
        id: &str,
        model: &str,
        tool_id: Option<&str>,
        name: Option<&str>,
        args: Option<&str>,
        finish_reason: Option<&str>,
    ) -> String {
        let mut tc = json!({ "index": 0 });
        if let Some(tid) = tool_id {
            tc["id"] = json!(tid);
            tc["type"] = json!("function");
        }
        let mut func = json!({});
        if let Some(n) = name {
            func["name"] = json!(n);
        }
        if let Some(a) = args {
            func["arguments"] = json!(a);
        }
        if !func.as_object().unwrap().is_empty() {
            tc["function"] = func;
        }
        let mut choice = json!({ "index": 0, "delta": { "tool_calls": [tc] } });
        if let Some(fr) = finish_reason {
            choice["finish_reason"] = json!(fr);
        }
        let chunk = json!({
            "id": id,
            "model": model,
            "choices": [choice],
        });
        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())
    }

    fn openai_chunk_with_tool_call_at_index(
        id: &str,
        model: &str,
        tool_index: usize,
        tool_id: Option<&str>,
        name: Option<&str>,
        args: Option<&str>,
        finish_reason: Option<&str>,
    ) -> String {
        let mut tc = json!({ "index": tool_index });
        if let Some(tid) = tool_id {
            tc["id"] = json!(tid);
            tc["type"] = json!("function");
        }
        let mut func = json!({});
        if let Some(n) = name {
            func["name"] = json!(n);
        }
        if let Some(a) = args {
            func["arguments"] = json!(a);
        }
        if !func.as_object().unwrap().is_empty() {
            tc["function"] = func;
        }
        let mut choice = json!({ "index": 0, "delta": { "tool_calls": [tc] } });
        if let Some(fr) = finish_reason {
            choice["finish_reason"] = json!(fr);
        }
        let chunk = json!({
            "id": id,
            "model": model,
            "choices": [choice],
        });
        format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap())
    }

    fn openai_done() -> String {
        "data: [DONE]\n\n".to_string()
    }

    fn make_stream(
        chunks: Vec<String>,
    ) -> impl futures::Stream<Item = Result<Bytes, TestError>> + Send + 'static {
        stream::iter(chunks.into_iter().map(|c| Ok(Bytes::from(c))))
    }

    async fn collect_events(chunks: Vec<String>, model: &str) -> Vec<Value> {
        let s = make_stream(chunks);
        let sse = create_sse_stream(s, model.to_string());
        tokio::pin!(sse);

        let mut events = Vec::new();
        while let Some(Ok(bytes)) = sse.next().await {
            let text = String::from_utf8_lossy(&bytes);
            for segment in text.split("\n\n").filter(|s| !s.is_empty()) {
                if let Some(data_line) = segment.lines().find(|l| l.starts_with("data: ")) {
                    let json_str = data_line.strip_prefix("data: ").unwrap();
                    if let Ok(v) = serde_json::from_str::<Value>(json_str) {
                        events.push(v);
                    }
                }
            }
        }
        events
    }

    #[tokio::test]
    async fn text_stream_produces_message_start_content_block_and_stop() {
        let chunks = vec![
            openai_chunk("chatcmpl-1", "gpt-4o", Some("Hello"), None),
            openai_chunk("chatcmpl-1", "gpt-4o", Some(" world"), None),
            openai_chunk("chatcmpl-1", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;

        assert_eq!(events[0]["type"], "message_start");
        assert_eq!(events[0]["message"]["id"], "chatcmpl-1");
        assert_eq!(events[0]["message"]["model"], "gpt-4o");
        assert_eq!(events[0]["message"]["role"], "assistant");

        assert_eq!(events[1]["type"], "content_block_start");
        assert_eq!(events[1]["content_block"]["type"], "text");

        assert_eq!(events[2]["type"], "content_block_delta");
        assert_eq!(events[2]["delta"]["type"], "text_delta");
        assert_eq!(events[2]["delta"]["text"], "Hello");

        assert_eq!(events[3]["type"], "content_block_delta");
        assert_eq!(events[3]["delta"]["text"], " world");

        assert_eq!(events[4]["type"], "content_block_stop");

        assert_eq!(events[5]["type"], "message_delta");
        assert_eq!(events[5]["delta"]["stop_reason"], "end_turn");

        assert_eq!(events[6]["type"], "message_stop");
    }

    #[tokio::test]
    async fn thinking_then_text_produces_two_content_blocks() {
        let chunks = vec![
            openai_chunk_with_reasoning("chatcmpl-2", "gpt-4o", "Let me think..."),
            openai_chunk_with_reasoning("chatcmpl-2", "gpt-4o", " more thinking"),
            openai_chunk("chatcmpl-2", "gpt-4o", Some("The answer is 42"), None),
            openai_chunk("chatcmpl-2", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;

        assert_eq!(events[0]["type"], "message_start");
        assert_eq!(events[1]["type"], "content_block_start");
        assert_eq!(events[1]["content_block"]["type"], "thinking");
        assert_eq!(events[1]["index"], 0);
        assert_eq!(events[4]["type"], "content_block_stop");
        assert_eq!(events[4]["index"], 0);
        assert_eq!(events[5]["type"], "content_block_start");
        assert_eq!(events[5]["content_block"]["type"], "text");
        assert_eq!(events[5]["index"], 1);
    }

    #[tokio::test]
    async fn tool_call_stream_produces_tool_use_block() {
        let chunks = vec![
            openai_chunk_with_tool_call(
                "chatcmpl-3",
                "gpt-4o",
                Some("call_abc"),
                Some("read_file"),
                None,
                None,
            ),
            openai_chunk_with_tool_call(
                "chatcmpl-3",
                "gpt-4o",
                None,
                None,
                Some("{\"path\":"),
                None,
            ),
            openai_chunk_with_tool_call(
                "chatcmpl-3",
                "gpt-4o",
                None,
                None,
                Some("\"/tmp\"}"),
                None,
            ),
            openai_chunk("chatcmpl-3", "gpt-4o", None, Some("tool_calls")),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;
        assert_eq!(events[1]["content_block"]["type"], "tool_use");
        assert_eq!(events[1]["content_block"]["id"], "call_abc");
        assert_eq!(events[5]["delta"]["stop_reason"], "tool_use");
    }

    #[tokio::test]
    async fn done_without_finish_reason_still_produces_message_stop() {
        let chunks = vec![
            openai_chunk("chatcmpl-4", "gpt-4o", Some("hi"), None),
            openai_done(),
        ];
        let events = collect_events(chunks, "fallback").await;
        assert_eq!(events.last().unwrap()["type"], "message_stop");
    }

    #[tokio::test]
    async fn fallback_model_used_when_upstream_omits_model() {
        let chunk = json!({
            "choices": [{ "index": 0, "delta": { "content": "hey" } }],
        });
        let chunks = vec![
            format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()),
            openai_chunk("id", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];
        let events = collect_events(chunks, "my-fallback-model").await;
        assert_eq!(events[0]["message"]["model"], "my-fallback-model");
    }

    #[tokio::test]
    async fn empty_content_chunks_are_not_emitted() {
        let chunks = vec![
            openai_chunk("chatcmpl-5", "gpt-4o", Some(""), None),
            openai_chunk("chatcmpl-5", "gpt-4o", Some("hello"), None),
            openai_chunk("chatcmpl-5", "gpt-4o", None, Some("stop")),
            openai_done(),
        ];
        let events = collect_events(chunks, "fallback").await;
        let text_deltas: Vec<_> = events
            .iter()
            .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta")
            .collect();
        assert_eq!(text_deltas.len(), 1);
        assert_eq!(text_deltas[0]["delta"]["text"], "hello");
    }

    #[tokio::test]
    async fn stream_error_produces_error_event_and_stops() {
        let items: Vec<Result<Bytes, TestError>> = vec![
            Ok(Bytes::from(openai_chunk(
                "chatcmpl-6",
                "gpt-4o",
                Some("start"),
                None,
            ))),
            Err(TestError),
        ];
        let s = stream::iter(items);
        let sse = create_sse_stream(s, "fallback".to_string());
        tokio::pin!(sse);

        let mut events = Vec::new();
        while let Some(Ok(bytes)) = sse.next().await {
            let text = String::from_utf8_lossy(&bytes);
            for segment in text.split("\n\n").filter(|s| !s.is_empty()) {
                if let Some(data_line) = segment.lines().find(|l| l.starts_with("data: ")) {
                    let json_str = data_line.strip_prefix("data: ").unwrap();
                    if let Ok(v) = serde_json::from_str::<Value>(json_str) {
                        events.push(v);
                    }
                }
            }
        }
        let error_events: Vec<_> = events.iter().filter(|e| e["type"] == "error").collect();
        assert_eq!(error_events.len(), 1);
    }

    #[tokio::test]
    async fn chunked_delivery_handles_split_sse_frames() {
        let full_chunk = openai_chunk("chatcmpl-7", "gpt-4o", Some("split"), None);
        let mid = full_chunk.len() / 2;
        let part1 = full_chunk[..mid].to_string();
        let part2 = format!(
            "{}{}{}",
            &full_chunk[mid..],
            openai_chunk("chatcmpl-7", "gpt-4o", None, Some("stop")),
            openai_done()
        );
        let events = collect_events(vec![part1, part2], "fallback").await;
        let text_deltas: Vec<_> = events
            .iter()
            .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta")
            .collect();
        assert_eq!(text_deltas.len(), 1);
        assert_eq!(text_deltas[0]["delta"]["text"], "split");
    }

    #[tokio::test]
    async fn text_then_tool_call_produces_two_blocks() {
        let chunks = vec![
            openai_chunk("chatcmpl-8", "gpt-4o", Some("I'll read that file."), None),
            openai_chunk_with_tool_call(
                "chatcmpl-8",
                "gpt-4o",
                Some("call_xyz"),
                Some("read_file"),
                None,
                None,
            ),
            openai_chunk_with_tool_call(
                "chatcmpl-8",
                "gpt-4o",
                None,
                None,
                Some("{\"path\":\"/etc\"}"),
                None,
            ),
            openai_chunk("chatcmpl-8", "gpt-4o", None, Some("tool_calls")),
            openai_done(),
        ];
        let events = collect_events(chunks, "fallback").await;
        let block_starts: Vec<_> = events
            .iter()
            .filter(|e| e["type"] == "content_block_start")
            .collect();
        assert_eq!(block_starts.len(), 2);
        assert_eq!(block_starts[0]["content_block"]["type"], "text");
        assert_eq!(block_starts[1]["content_block"]["type"], "tool_use");
    }

    // ── Scenario A (fixed): Tool call index mapping prevents misassignment ──
    //
    // Previously, `emit_tool_calls` ignored `DeltaToolCall.index` and used its
    // own `next_index` counter.  When arguments for index0 arrived after
    // index1 had been opened, they were emitted at the WRONG block (index1).
    //
    // After fix:  a `tool_call_map` tracks each upstream index → block index.
    // Arguments for an already-closed index are dropped instead of misassigned.
    // ─────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn scenario_a_tool_call_args_not_misassigned_on_interleave() {
        // Upstream sends args for idx0 AFTER idx1 opened the block.
        // Fix: these args are silently dropped (Anthropic protocol forbids
        // appending to a closed block) instead of going to idx1.
        let chunks = vec![
            openai_chunk_with_tool_call_at_index(
                "chatcmpl-99", "m", 0,
                Some("call_read"), Some("read_file"), None, None,
            ),
            openai_chunk_with_tool_call_at_index(
                "chatcmpl-99", "m", 1,
                Some("call_search"), Some("search_web"), None, None,
            ),
            // These args for idx0 arrive too late — block0 already closed
            openai_chunk_with_tool_call_at_index(
                "chatcmpl-99", "m", 0,
                None, None, Some(r#"{"path":"/etc"}"#), None,
            ),
            openai_chunk_with_tool_call_at_index(
                "chatcmpl-99", "m", 1,
                None, None, Some(r#"{}"#), None,
            ),
            openai_chunk("chatcmpl-99", "m", None, Some("tool_calls")),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;

        // Collect block info
        let block_starts: Vec<_> = events.iter()
            .filter(|e| e["type"] == "content_block_start")
            .enumerate()
            .map(|(i, e)| (i, e["index"].as_u64().unwrap(), e["content_block"]["id"].as_str().unwrap().to_string()))
            .collect();
        assert_eq!(block_starts.len(), 2);

        // Block0 = call_read, Block1 = call_search
        assert_eq!(block_starts[0].1, 0); // index0
        assert_eq!(block_starts[0].2, "call_read");
        assert_eq!(block_starts[1].1, 1); // index1
        assert_eq!(block_starts[1].2, "call_search");

        // Verify NO input_json_delta at index0 (call_read's args were dropped)
        let read_deltas = events.iter()
            .filter(|e| e["type"] == "content_block_delta"
                && e["index"] == 0
                && e["delta"]["type"] == "input_json_delta")
            .count();
        assert_eq!(read_deltas, 0,
            "call_read (index0) must have zero input_json deltas — interleaved args dropped, not misassigned");

        // Verify only ONE input_json_delta at index1 (call_search got its own args)
        let search_deltas = events.iter()
            .filter(|e| e["type"] == "content_block_delta"
                && e["delta"]["type"] == "input_json_delta")
            .count();
        assert_eq!(search_deltas, 1,
            "call_search (index1) must have exactly one input_json delta");

        // Each block properly started, stopped, and message_delta+stop present
        assert_eq!(events.last().unwrap()["type"], "message_stop");
    }

    // ── Scenario A (correct-order): Normal sequential tool calls ───────────
    //
    // When tool call arguments arrive in order (idx0 id → idx0 args → idx1 id
    // → idx1 args), the fix must NOT break the common case.
    // ─────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn scenario_a_tool_call_args_correct_order_still_works() {
        let chunks = vec![
            // idx0 id+name
            openai_chunk_with_tool_call_at_index(
                "chatcmpl-99", "m", 0,
                Some("call_0"), Some("read_file"), None, None,
            ),
            // idx0 args
            openai_chunk_with_tool_call_at_index(
                "chatcmpl-99", "m", 0,
                None, None, Some(r#"{"path":"/tmp"}"#), None,
            ),
            // idx1 id+name
            openai_chunk_with_tool_call_at_index(
                "chatcmpl-99", "m", 1,
                Some("call_1"), Some("search_web"), None, None,
            ),
            // idx1 args
            openai_chunk_with_tool_call_at_index(
                "chatcmpl-99", "m", 1,
                None, None, Some(r#"{}"#), None,
            ),
            openai_chunk("chatcmpl-99", "m", None, Some("tool_calls")),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;

        // Block0: call_read at index0 gets its args
        let block0_deltas: Vec<_> = events.iter()
            .filter(|e| e["type"] == "content_block_delta"
                && e["index"] == 0)
            .collect();
        assert_eq!(block0_deltas.len(), 1,
            "call_read (index0) must get its args in correct order");
        assert_eq!(block0_deltas[0]["delta"]["partial_json"], r#"{"path":"/tmp"}"#);

        // Block1: call_search at index1 gets its args
        let block1_deltas: Vec<_> = events.iter()
            .filter(|e| e["type"] == "content_block_delta"
                && e["index"] == 1)
            .collect();
        assert_eq!(block1_deltas.len(), 1,
            "call_search (index1) must get its args in correct order");
    }

    // ── Scenario B (fixed): content after finish_reason gets a NEW block ──
    //
    // Previously, `close_current_block` did NOT reset `state.block` to Idle.
    // Content arriving after finish_reason was emitted as a delta on the
    // ALREADY-STOPPED block (wrong index).  Then `translate_done` emitted
    // `message_stop` without closing the block or emitting `message_delta`.
    //
    // After fix:
    //   1. close_current_block resets state.block → content opens a NEW block
    //   2. translate_done closes open blocks before message_stop
    // ─────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn scenario_b_content_after_finish_gets_new_block() {
        let chunks = vec![
            openai_chunk("chatcmpl-100", "m", Some("Hello"), None),
            openai_chunk("chatcmpl-100", "m", None, Some("stop")),
            openai_chunk("chatcmpl-100", "m", Some(" world"), None),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;
        let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();

        // After the first content_block_stop(0), a NEW block opens at index1
        let stop0 = types.iter().position(|t| *t == "content_block_stop").unwrap();
        let block_starts_after: Vec<_> = events[stop0 + 1..].iter()
            .filter(|e| e["type"] == "content_block_start")
            .collect();
        assert_eq!(block_starts_after.len(), 1,
            "must open a NEW content block for content arriving after finish_reason");
        assert_eq!(block_starts_after[0]["index"], 1,
            "the new block must have a fresh index (1), not the old (0)");

        // The late " world" delta lands at index1, NOT index0
        let late_delta: Vec<_> = events.iter()
            .filter(|e| e["type"] == "content_block_delta" && e["delta"]["text"] == " world")
            .collect();
        assert_eq!(late_delta.len(), 1);
        assert_eq!(late_delta[0]["index"], 1,
            "' world' delta must be at the NEW block index (1), not the old closed block");

        // Stream ends properly: content_block_stop → message_stop
        assert_eq!(events.last().unwrap()["type"], "message_stop");
        assert_eq!(types.last().unwrap(), &"message_stop");
    }

    // ── Scenario C (fixed): duplicate finish_reason is idempotent ──────────
    //
    // Previously, `emit_finish` always emitted message_delta and
    // close_current_block could emit duplicate content_block_stop because
    // state.block was never reset.
    //
    // After fix:
    //   1. emit_finish sets state.finished=true and subsequent calls return early
    //   2. close_current_block resets state.block → second close is a no-op
    // ─────────────────────────────────────────────────────────────────────
    #[tokio::test]
    async fn scenario_c_duplicate_finish_reason_is_idempotent() {
        let chunks = vec![
            openai_chunk("chatcmpl-101", "m", Some("Hello"), None),
            openai_chunk("chatcmpl-101", "m", None, Some("stop")),
            // second identical finish_reason — should be a no-op
            openai_chunk("chatcmpl-101", "m", None, Some("stop")),
            openai_done(),
        ];

        let events = collect_events(chunks, "fallback").await;
        let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();

        // Only ONE content_block_stop
        let stop_count = types.iter().filter(|t| **t == "content_block_stop").count();
        assert_eq!(stop_count, 1,
            "duplicate finish_reason must NOT produce a second content_block_stop");

        // Only ONE message_delta
        let delta_count = types.iter().filter(|t| **t == "message_delta").count();
        assert_eq!(delta_count, 1,
            "duplicate finish_reason must NOT produce a second message_delta");

        // Stream still ends with message_stop
        assert_eq!(events.last().unwrap()["type"], "message_stop");

        // Total event sequence is the canonical 6-event form
        assert_eq!(types, &["message_start", "content_block_start",
            "content_block_delta", "content_block_stop",
            "message_delta", "message_stop"],
            "duplicate finish_reason must produce exactly the same events as a single finish");
    }
}
