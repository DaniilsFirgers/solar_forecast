use std::sync::Arc;

use lazy_static::lazy_static;
use mongodb::bson::Document;
use tokio::sync::{mpsc, Mutex};

lazy_static! {
    pub static ref CHANNEL: Channel = Channel::new();
}

pub struct Channel {
    sender: mpsc::UnboundedSender<Vec<Document>>,
    receiver: Arc<Mutex<mpsc::UnboundedReceiver<Vec<Document>>>>,
}

impl Channel {
    fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel::<Vec<Document>>();
        Self {
            sender,
            receiver: Arc::new(Mutex::new(receiver)),
        }
    }
    pub fn document_sender(&self) -> mpsc::UnboundedSender<Vec<Document>> {
        self.sender.clone()
    }

    pub fn document_receiver(&self) -> Arc<Mutex<mpsc::UnboundedReceiver<Vec<Document>>>> {
        self.receiver.clone()
    }
}
