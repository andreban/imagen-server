# Builder
FROM rust:1 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

# Runner
FROM debian:bookworm-slim as runner
RUN apt update
RUN apt install openssl ca-certificates -y

COPY --from=builder /app/target/release/imagen-server /usr/local/bin/imagen-server
COPY --from=builder /app/static /static

CMD ["imagen-server"]
