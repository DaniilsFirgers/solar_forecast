use std::env;

use env_logger;

pub fn init_logger() {
    env::set_var("RUST_LOG", "trace");
    env_logger::init();
}
