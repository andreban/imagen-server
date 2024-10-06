use std::{env, error::Error, sync::Arc};

use axum::{
    body::Body,
    extract::{Path, Query, Request, State},
    http::{header, StatusCode},
    middleware::{from_fn, Next},
    response::Response,
    routing::{get, post},
    Form, Router,
};
use gcp_auth::TokenProvider;
use gemini_rs::prelude::{
    GeminiClient, PredictImageRequest, PredictImageRequestParameters,
    PredictImageRequestParametersOutputOptions, PredictImageRequestPrompt,
};
use serde::Deserialize;
use tower_http::{compression::CompressionLayer, services::ServeDir};
use tracing::info;

const IMAGEN_MODEL: &str = "imagen-3.0-fast-generate-001";

#[derive(Clone)]
pub struct AppState {
    pub vertex_client: GeminiClient<Arc<dyn TokenProvider>>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt().init();

    let api_endpoint = env::var("API_ENDPOINT")?;
    let project_id = env::var("PROJECT_ID")?;
    let location_id = env::var("LOCATION_ID")?;
    let bind_address = env::var("BIND_ADDRESS").unwrap_or_else(|_| "127.0.0.1:8080".to_string());

    let authentication_manager = gcp_auth::provider().await?;
    tracing::info!("GCP AuthenticationManager initialized.");

    let vertex_client = GeminiClient::new(
        authentication_manager,
        api_endpoint,
        project_id,
        location_id,
    );

    let app_state = AppState {
        vertex_client: vertex_client,
    };

    let listener = tokio::net::TcpListener::bind(bind_address).await?;

    // Sets up a compression layer that supports brotli, deflate, gzip, and zstd.
    let compression_layer: CompressionLayer = CompressionLayer::new()
        .br(true)
        .deflate(true)
        .gzip(true)
        .zstd(true);

    let app = Router::new()
        .fallback_service(ServeDir::new("static"))
        .route("/generate", post(generate_image_post))
        .route("/generate/:path", get(generate_image_path))
        .layer(from_fn(cache_control))
        .layer(compression_layer)
        .with_state(app_state);

    tracing::info!(addr = listener.local_addr()?.to_string(), "Server started",);
    axum::serve(listener, app).await?;

    Ok(())
}

pub async fn cache_control(req: Request, next: Next) -> Response {
    let mut res = next.run(req).await;
    res.headers_mut().insert(
        header::CACHE_CONTROL,
        header::HeaderValue::from_static("no-store"),
    );
    res
}

#[derive(Debug, Deserialize)]
struct GenerateImageParams {
    pub prompt: String,
    pub mime_type: String,
    pub aspect_ratio: String,
}

async fn generate_image_post(
    State(app_state): State<AppState>,
    Form(params): Form<GenerateImageParams>,
) -> Response<Body> {
    info!("Generating image with params: {:?}", params);

    generate_image(
        app_state.vertex_client.clone(),
        params.prompt,
        params.mime_type,
        params.aspect_ratio,
    )
    .await
}

#[derive(Debug, Deserialize)]
struct GenerateImagePathParams {
    pub mime_type: String,
    pub aspect_ratio: String,
}

async fn generate_image_path(
    State(app_state): State<AppState>,
    Path(path): Path<String>,
    Query(params): Query<GenerateImagePathParams>,
) -> Response<Body> {
    info!("Generating image with params: {:?}", params);

    generate_image(
        app_state.vertex_client.clone(),
        path,
        params.mime_type,
        params.aspect_ratio,
    )
    .await
}

async fn generate_image(
    vertex_client: GeminiClient<Arc<dyn TokenProvider>>,
    prompt: String,
    mime_type: String,
    aspect_ratio: String,
) -> Response<Body> {
    let request = PredictImageRequest {
        instances: vec![PredictImageRequestPrompt { prompt }],
        parameters: PredictImageRequestParameters {
            aspect_ratio: Some(aspect_ratio),
            output_options: Some(PredictImageRequestParametersOutputOptions {
                mime_type: Some(mime_type),
                compression_quality: Some(75),
            }),
            sample_count: 1,
            ..Default::default()
        },
    };

    let mut response = match vertex_client.predict_image(&request, IMAGEN_MODEL).await {
        Ok(response) => response,
        Err(e) => {
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from(format!("Error: {}", e)))
                .unwrap()
        }
    };

    let result = match response.predictions.pop() {
        Some(result) => result,
        None => {
            return Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from("No predictions returned"))
                .unwrap()
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", result.mime_type)
        .body(Body::from(result.bytes_base64_encoded))
        .unwrap()
}
