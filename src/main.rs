#![feature(box_patterns)]

mod lexer;
mod parser;

use crate::lexer::Lexer;
use std::env;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <filename>", args[0]);
        std::process::exit(1);
    }

    let filename = &args[1];
    let source = fs::read_to_string(filename)?;

    let tokens = Lexer::<'_>::tokenize(&source)?;
    let ast = parser::parse(tokens)?;

    println!("AST: {:?}", ast);

    // TODO: Implement type checking and transpilation
    // type_checker::check(&ast)?;
    // let rust_code = transpiler::transpile(&ast)?;
    // println!("Generated Rust code:\n{}", rust_code);

    Ok(())
}
