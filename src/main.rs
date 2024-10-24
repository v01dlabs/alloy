#![feature(box_patterns)]

mod error;
mod lexer;
mod parser;
mod type_checker;
mod ast;
mod ty;

use crate::error::CompilerError;
use crate::lexer::Lexer;
use std::env;
use std::error::Error;
use std::fs;


fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <filename>", args[0]);
        std::process::exit(1);
    }

    let filename = &args[1];
    let source = fs::read_to_string(filename)?;

    let tokens = Lexer::<'_>::tokenize(&source).map_err(CompilerError::LexerError)?;
    let ast = parser::parse(tokens).map_err(CompilerError::ParserError)?;

    println!("AST: {:?}", ast);

    // TODO: Implement type checking and transpilation
    type_checker::typecheck(&ast)?;
    // let rust_code = transpiler::transpile(&ast)?;
    // println!("Generated Rust code:\n{}", rust_code);

    Ok(())
}
