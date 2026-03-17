use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{info, warn};

use crate::channels::base::Channel;
use crate::core::agent_loop::AgentLoop;
use crate::core::config::{AppConfig, ExecChannelConfig};
use crate::core::error::{CrabClawError, Result};

pub struct ExecChannel {
    config: Arc<AppConfig>,
    workspace: PathBuf,
    exec_config: ExecChannelConfig,
    channel_name: String,
    child: Option<tokio::process::Child>,
}

impl ExecChannel {
    pub fn new(config: Arc<AppConfig>, workspace: PathBuf, exec_config: ExecChannelConfig) -> Self {
        let channel_name = format!("exec:{}", exec_config.name);
        Self {
            config,
            workspace,
            exec_config,
            channel_name,
            child: None,
        }
    }
}

#[async_trait]
impl Channel for ExecChannel {
    fn name(&self) -> &str {
        &self.channel_name
    }

    async fn start(&mut self) -> Result<()> {
        let output_dir = self
            .workspace
            .join(".crabclaw")
            .join("exec")
            .join(&self.exec_config.name);
        tokio::fs::create_dir_all(&output_dir)
            .await
            .map_err(CrabClawError::Io)?;

        // Build merged config with exec-specific prompt appended
        let mut merged = (*self.config).clone();
        if let Some(ref extra) = self.exec_config.prompt {
            merged.system_prompt = Some(match merged.system_prompt {
                Some(ref existing) => format!("{existing}\n{extra}"),
                None => extra.clone(),
            });
        }
        let history_messages = self.exec_config.history_messages.unwrap_or(0);

        // Spawn the external command
        let mut child = tokio::process::Command::new("/bin/sh")
            .args(["-c", &self.exec_config.command])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(CrabClawError::Io)?;

        let stdout = child.stdout.take().expect("stdout is piped");
        self.child = Some(child);

        let mut lines = BufReader::new(stdout).lines();

        while let Some(line) = lines.next_line().await.map_err(CrabClawError::Io)? {
            let json_val: serde_json::Value = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    warn!("exec.{}.parse_error: {e}", self.exec_config.name);
                    continue;
                }
            };

            let session_id = format!("exec:{}:ephemeral", self.exec_config.name);
            let mut agent =
                match AgentLoop::open(&merged, &self.workspace, &session_id, history_messages) {
                    Ok(a) => a,
                    Err(e) => {
                        warn!("exec.{}.agent_loop.error: {e}", self.exec_config.name);
                        continue;
                    }
                };

            let result = agent.handle_input(&json_val.to_string()).await;

            // The history window controls model-visible context; reset_tape only
            // keeps the ephemeral exec tape from growing without bound on disk.
            if let Err(e) = agent.reset_tape() {
                warn!("exec.{}.reset_tape.error: {e}", self.exec_config.name);
            }

            if let Some(reply) = result.to_reply() {
                let filename = make_output_filename();
                let out_path = output_dir.join(&filename);
                if let Err(e) = tokio::fs::write(&out_path, reply).await {
                    warn!("exec.{}.write.error: {e}", self.exec_config.name);
                }
            }
        }

        info!("exec.{}.stdout_closed", self.exec_config.name);
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        if let Some(child) = self.child.as_mut() {
            let _ = child.kill().await;
        }
        Ok(())
    }
}

fn make_output_filename() -> String {
    let ts = chrono::Utc::now().format("%Y-%m-%dT%H%M%S%.9f");
    let rand: u16 = rand::random();
    format!("{ts}_{rand:04x}.md")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_channel_name() {
        let config = Arc::new(AppConfig {
            profile: String::new(),
            api_key: String::new(),
            anthropic_access_token: None,
            api_base: String::new(),
            model: String::new(),
            system_prompt: None,
            telegram_token: None,
            telegram_allow_from: vec![],
            telegram_allow_chats: vec![],
            telegram_proxy: None,
            max_context_messages: 50,
            exec_channels: vec![],
        });
        let exec_cfg = ExecChannelConfig {
            name: "foo".to_string(),
            command: "echo test".to_string(),
            prompt: None,
            history_messages: None,
        };
        let ch = ExecChannel::new(config, PathBuf::from("/tmp"), exec_cfg);
        assert_eq!(ch.name(), "exec:foo");
    }

    #[test]
    fn output_filename_format() {
        let f = make_output_filename();
        assert!(f.ends_with(".md"));
        // Format: <timestamp>_<4hex>.md — at least 8 chars before .md
        assert!(f.len() > 8);
    }

    #[test]
    fn parse_json_line_valid() {
        let line = r#"{"id":1,"msg":"hello"}"#;
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        assert_eq!(v["id"], 1);
    }

    #[test]
    fn parse_json_line_invalid() {
        let line = "not-json";
        let result: serde_json::Result<serde_json::Value> = serde_json::from_str(line);
        assert!(result.is_err());
    }
}
