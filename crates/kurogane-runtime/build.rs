fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Vendored rather than requiring a system `protoc` install, so `cargo
    // build` works the same on any machine without extra setup. Configured
    // directly on prost_build::Config rather than the PROTOC env var, since
    // this workspace forbids unsafe code and std::env::set_var is unsafe.
    let mut config = tonic_prost_build::Config::new();
    config.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);

    tonic_prost_build::configure().compile_with_config(
        config,
        &["proto/kurogane.proto"],
        &["proto"],
    )?;
    Ok(())
}
