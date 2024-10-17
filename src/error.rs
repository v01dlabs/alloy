use std::error::Error;
use std::fmt;

use crate::lexer::Location;



#[derive(Debug)]
pub enum CompilerError {
    LexerError(LexerError),
    ParserError(ParserError),
}

impl fmt::Display for CompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilerError::LexerError(e) => write!(f, "Lexer error: {}", e),
            CompilerError::ParserError(e) => write!(f, "Parser error: {}", e),
        }
    }
}

impl Error for CompilerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            CompilerError::LexerError(e) => Some(e),
            CompilerError::ParserError(e) => Some(e),
        }
    }
}

#[derive(Debug)]
pub enum LexerError {
    UnexpectedChar(char, Location),
    InvalidNumber(String, Location),
    UnterminatedString(Location),
}

impl LexerError {
    /// Returns the location of the error in the source code.
    fn location(&self) -> Location {
        match self {
            LexerError::UnexpectedChar(_, l) => *l,
            LexerError::InvalidNumber(_, l) => *l,
            LexerError::UnterminatedString(l) => *l,
        }
    }
}

impl fmt::Display for LexerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexerError::UnexpectedChar(c, l) => write!(f, "Unexpected character at {}: {}", l, c),
            LexerError::InvalidNumber(s, l) => write!(f, "Invalid number at {}: {}", l, s),
            LexerError::UnterminatedString(l) => write!(f, "Unterminated string at {}", l),
        }
    }
}

impl Error for LexerError {}

#[derive(Debug)]
pub enum ParserError {
    UnexpectedToken(String),
    ExpectedToken(String, String),
    InvalidExpression,
}

impl fmt::Display for ParserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParserError::UnexpectedToken(msg) => write!(f, "Unexpected token: {}", msg),
            ParserError::ExpectedToken(expected, found) => {
                write!(f, "Expected {}, found {}", expected, found)
            }
            ParserError::InvalidExpression => write!(f, "Invalid expression"),
        }
    }
}

impl Error for ParserError {}
