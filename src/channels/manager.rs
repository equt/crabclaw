use std::sync::Arc;

use tracing::info;

use crate::channels::base::Channel;
use crate::channels::exec::ExecChannel;
use crate::channels::telegram::TelegramChannel;
use crate::core::config::AppConfig;
use crate::core::error::{CrabClawError, Result};

/// Manages channel lifecycles.
///
/// Aligned with bub's `ChannelManager`:
/// - Registers enabled channels based on config
/// - Runs all channels concurrently
pub struct ChannelManager {
    channels: Vec<Box<dyn Channel>>,
}

impl ChannelManager {
    pub fn new(config: Arc<AppConfig>, workspace: &std::path::Path) -> Self {
        let mut channels: Vec<Box<dyn Channel>> = Vec::new();

        if config.telegram_enabled() {
            info!("channel_manager.register: telegram");
            channels.push(Box::new(TelegramChannel::new(
                Arc::clone(&config),
                workspace.to_path_buf(),
            )));
        }

        for exec_cfg in config.exec_channels.iter() {
            info!("channel_manager.register: exec:{}", exec_cfg.name);
            channels.push(Box::new(ExecChannel::new(
                Arc::clone(&config),
                workspace.to_path_buf(),
                exec_cfg.clone(),
            )));
        }

        Self { channels }
    }

    pub fn enabled_channels(&self) -> Vec<&str> {
        self.channels.iter().map(|c| c.name()).collect()
    }

    /// Run all registered channels concurrently. Blocks until all channels complete or error.
    pub async fn run(&mut self) -> Result<()> {
        if self.channels.is_empty() {
            return Err(CrabClawError::Config(
                "no channels enabled; set TELEGRAM_TOKEN or EXEC_CHANNELS to enable channels"
                    .to_string(),
            ));
        }

        info!(
            "channel_manager.start channels={:?}",
            self.enabled_channels()
        );

        let channels = std::mem::take(&mut self.channels);
        let mut handles = Vec::new();
        for mut channel in channels {
            handles.push(tokio::spawn(async move { channel.start().await }));
        }

        for handle in handles {
            handle
                .await
                .map_err(|e| CrabClawError::Config(format!("channel task panicked: {e}")))??;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(telegram_token: Option<&str>) -> Arc<AppConfig> {
        Arc::new(AppConfig {
            profile: "test".to_string(),
            api_key: "key".to_string(),
            anthropic_access_token: None,
            api_base: "https://api.example.com".to_string(),
            model: "openai:test-model".to_string(),
            system_prompt: None,
            telegram_token: telegram_token.map(String::from),
            telegram_allow_from: vec![],
            telegram_allow_chats: vec![],
            telegram_proxy: None,
            max_context_messages: 50,
            exec_channels: vec![],
        })
    }

    #[test]
    fn no_channels_when_token_missing() {
        let config = test_config(None);
        let mgr = ChannelManager::new(config, std::path::Path::new("/tmp"));
        assert!(mgr.enabled_channels().is_empty());
    }

    #[test]
    fn telegram_registered_when_token_set() {
        let config = test_config(Some("test-token"));
        let mgr = ChannelManager::new(config, std::path::Path::new("/tmp"));
        assert_eq!(mgr.enabled_channels(), vec!["telegram"]);
    }

    #[test]
    fn exec_registered_for_each_named_channel() {
        use crate::core::config::ExecChannelConfig;
        let config = Arc::new(AppConfig {
            profile: "test".to_string(),
            api_key: "key".to_string(),
            anthropic_access_token: None,
            api_base: "https://api.example.com".to_string(),
            model: "openai:test-model".to_string(),
            system_prompt: None,
            telegram_token: None,
            telegram_allow_from: vec![],
            telegram_allow_chats: vec![],
            telegram_proxy: None,
            max_context_messages: 50,
            exec_channels: vec![
                ExecChannelConfig {
                    name: "foo".to_string(),
                    command: "echo ok".to_string(),
                    prompt: None,
                    history_messages: None,
                },
                ExecChannelConfig {
                    name: "bar".to_string(),
                    command: "echo ok".to_string(),
                    prompt: None,
                    history_messages: None,
                },
            ],
        });
        let mgr = ChannelManager::new(config, std::path::Path::new("/tmp"));
        let channels = mgr.enabled_channels();
        assert!(channels.contains(&"exec:foo"));
        assert!(channels.contains(&"exec:bar"));
        assert_eq!(channels.len(), 2);
    }
}
