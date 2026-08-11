use std::io::Read;

use clap::Args;

#[derive(Args, Debug)]
pub struct SolveLiveArgs {
    /// UTF-8 JSON request file. Omit or pass '-' to read the request from stdin.
    #[arg(long, default_value = "-")]
    pub request_json: String,
}

pub fn execute(args: &SolveLiveArgs) {
    let request = if args.request_json == "-" {
        let mut bytes = Vec::new();
        std::io::stdin().read_to_end(&mut bytes).unwrap();
        bytes
    } else {
        std::fs::read(&args.request_json).unwrap()
    };
    println!("{}", donatello_ffi::solve_json(&request));
}
