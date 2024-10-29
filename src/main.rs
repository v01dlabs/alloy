#![feature(box_patterns)]

mod ast;
mod error;
mod lexer;
mod parser;

mod transpiler;
mod type_checker;

use crate::error::CompilerError;
use crate::lexer::Lexer;
use std::{env, error::Error, fs, process::exit};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <filename>", args[0]);
        exit(1);
    }

    let filename = &args[1];
    let source = fs::read_to_string(filename)?;

    let tokens = Lexer::<'_>::tokenize(&source).map_err(CompilerError::Lexer)?;
    let ast = parser::parse(tokens).map_err(CompilerError::Parser)?;
    let ast = *ast;

    println!("AST: {:?}", ast);

    type_checker::typecheck(&ast).map_err(CompilerError::Type)?;
    let rust_code = transpiler::transpile(&ast).map_err(CompilerError::Transpiler)?;
    println!("Generated Rust code:\n{}", rust_code);

    Ok(())
}
