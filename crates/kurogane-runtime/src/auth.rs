//! Authenticated peer identity, kept deliberately boring: a shared
//! cluster-wide token in gRPC metadata, not TLS/mTLS/PKI — that's real
//! production hardening this learning project doesn't need yet. This proves
//! "the caller is a legitimate cluster process"; `kurogane-raft`'s existing
//! `is_member` check (on the claimed `NodeId`) proves which member it is.

use tonic::metadata::MetadataValue;
use tonic::service::Interceptor;
use tonic::{Request, Status};

pub const TOKEN_METADATA_KEY: &str = "x-cluster-token";

/// Rejects any request whose `x-cluster-token` metadata doesn't match the
/// configured cluster token, before it ever reaches a service handler.
#[derive(Clone)]
pub struct TokenInterceptor {
    token: String,
}

impl TokenInterceptor {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }
}

impl Interceptor for TokenInterceptor {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        let presented = request
            .metadata()
            .get(TOKEN_METADATA_KEY)
            .ok_or_else(|| Status::unauthenticated("missing cluster token"))?;

        if presented.as_bytes() == self.token.as_bytes() {
            Ok(request)
        } else {
            Err(Status::unauthenticated("invalid cluster token"))
        }
    }
}

/// Attaches the cluster token to an outbound request, the client-side half
/// of `TokenInterceptor`'s check.
pub fn attach_token<M>(mut request: Request<M>, token: &str) -> Request<M> {
    let value =
        MetadataValue::try_from(token).expect("configured cluster token is valid ASCII metadata");
    request.metadata_mut().insert(TOKEN_METADATA_KEY, value);
    request
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_request_with_no_token() {
        let mut interceptor = TokenInterceptor::new("secret");

        let result = interceptor.call(Request::new(()));

        assert_eq!(
            result.expect_err("missing token must be rejected").code(),
            tonic::Code::Unauthenticated
        );
    }

    #[test]
    fn rejects_a_request_with_the_wrong_token() {
        let mut interceptor = TokenInterceptor::new("secret");
        let request = attach_token(Request::new(()), "not-the-secret");

        let result = interceptor.call(request);

        assert_eq!(
            result.expect_err("wrong token must be rejected").code(),
            tonic::Code::Unauthenticated
        );
    }

    #[test]
    fn accepts_a_request_with_the_correct_token() {
        let mut interceptor = TokenInterceptor::new("secret");
        let request = attach_token(Request::new(()), "secret");

        assert!(interceptor.call(request).is_ok());
    }
}
