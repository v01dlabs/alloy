use std::fmt;


/// Represents a token in Alloy.
#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    // Keywords
    Let,
    Run,
    Return,
    Fn,
    If,
    
    Else,
    While,
    Loop,
    Do, // I really miss these in Rust sometimes
    For,
    Match,
    Guard,
    
    Mut,
    Async,
    Shared,
    Default,
    In,
    With,
    Where,
    
    Await,
    
    Struct, // ? unsure if keeping
    Enum, // ? unsure if keeping
    Union,
    Type,
    Effect,
    Trait,
    Handler,

    Impl,

    // Literals
    Identifier(String),
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),

    // Operators
    Plus,
    Minus,
    Multiply,
    Divide,
    Assign,
    Eq,
    Modulo,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
    Not,
    Pipeline,
    Increment,
    Decrement,
    Range,

    Typeof,
    As,

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Pipe,
    Comma,
    Dot,
    QuestionMark,
    ExclamationPt,
    Colon,
    PathSep,
    Arrow,
    Semicolon,
    Newline,
    

    // Special
    Eof,
}

impl Token {
    pub fn is_block_end(&self) -> bool {
        matches!(
            self,
            Token::RBrace | Token::RBracket | Token::RParen | Token::Eof
        )
    }

    pub fn is_block_start(&self) -> bool {
        matches!(self, Token::LBrace | Token::LBracket | Token::LParen)
            || matches!(
                self,
                Token::If
                    | Token::Let
                    | Token::Else
                    | Token::For
                    | Token::While
                    | Token::Guard
                    | Token::Match
            )
    }

    pub fn ident_to_keyword<'a>(&'a self) -> &'a Token {
        match self {
            Token::Identifier(ref ident) => match ident.as_str() {
                "handler" => &Token::Handler,
                "shared" => &Token::Shared,
                "default" => &Token::Default,
                "effect" => &Token::Effect,
                _ => self
            },
            _ => self
        }
    }

    pub fn is_punct(&self) -> bool {
        match self {
            Token::Plus
            | Token::Minus
            | Token::Multiply
            | Token::Divide
            | Token::Assign
            | Token::Eq
            | Token::Modulo
            | Token::NotEq
            | Token::Lt
            | Token::LtEq
            | Token::Gt
            | Token::GtEq
            | Token::And
            | Token::Or
            | Token::Not
            | Token::Pipe
            | Token::Comma
            | Token::Dot
            | Token::QuestionMark
            | Token::ExclamationPt
            | Token::Colon
            | Token::PathSep
            | Token::Arrow
            | Token::Semicolon
            | Token::Pipeline
            | Token::Increment
            | Token::Decrement
            | Token::Range => true,
            _ => false,
        }
    }

    pub fn can_begin_expr(&self) -> bool {
        match self {
            | Token::Identifier(_)
            | Token::IntLiteral(_)
            | Token::FloatLiteral(_)
            | Token::StringLiteral(_)
            | Token::BoolLiteral(_)
            | Token::LParen
            | Token::LBracket
            | Token::LBrace
            | Token::Not
            | Token::Minus
            | Token::Pipeline => true,
            _ => false,
        }   
    }

    pub fn can_begin_pattern(&self) -> bool {
        match self {
            | Token::Identifier(_)
            | Token::IntLiteral(_)
            | Token::FloatLiteral(_)
            | Token::StringLiteral(_)
            | Token::BoolLiteral(_)
            | Token::LParen
            | Token::LBracket
            | Token::Not
            | Token::Minus
            | Token::Pipe
            | Token::Range
            | Token::Pipeline => true,
            _ => false,
        }   
    }

    pub fn can_begin_type(&self) -> bool {
        match self {
            | Token::Identifier(_)
            | Token::LParen
            | Token::LBracket
            | Token::Not
            | Token::ExclamationPt
            | Token::QuestionMark
            | Token::Minus
            | Token::Pipe
            | Token::PathSep => true,
            _ => false,
        }   
    }

    pub fn can_begin_item(&self) -> bool {
        match self {
            Token::Fn
            | Token::Let
            | Token::Impl
            | Token::Struct
            | Token::Enum
            | Token::Union
            | Token::Trait
            | Token::Effect
            | Token::Handler => true,
            _ => false,
        }
    }

    pub fn is_literal(&self) -> bool {
        match self {
            Token::IntLiteral(_)
            | Token::FloatLiteral(_)
            | Token::StringLiteral(_)
            | Token::BoolLiteral(_) => true,
            _ => false,
        }
    }
    pub fn is_keyword(&self) -> bool {
        match self {
            Token::Let
            | Token::Fn
            | Token::If
            | Token::Else
            | Token::While
            | Token::Loop
            | Token::For
            | Token::Match
            | Token::Guard
            | Token::Mut
            | Token::Async
            | Token::Shared
            | Token::Default
            | Token::In
            | Token::With
            | Token::Where
            | Token::Await
            | Token::Struct
            | Token::Enum
            | Token::Union
            | Token::Type
            | Token::Effect
            | Token::Trait
            | Token::Handler
            | Token::Impl => true,
            Token::Identifier(_) => {
                let tok = self.clone();
                let tok = tok.ident_to_keyword();
                // if it changed, it's a keyword
                tok != self
            }
            _ => false,
        }
    }
}


impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {        
        match self {
            Token::Let => f.write_str("let"),
            Token::Run => f.write_str("run"),
            Token::Return => f.write_str("return"),
            Token::Fn => f.write_str("fn"),
            Token::If => f.write_str("if"),
            Token::Else => f.write_str("else"),
            Token::While => f.write_str("while"),
            Token::Loop => f.write_str("loop"),
            Token::Do => f.write_str("do"),
            Token::For => f.write_str("for"),
            Token::Match => f.write_str("match"),
            Token::Guard => f.write_str("guard"),
            Token::Mut => f.write_str("mut"),
            Token::Async => f.write_str("async"),
            Token::Shared => f.write_str("shared"),
            Token::Default => f.write_str("default"),
            Token::In => f.write_str("in"),
            Token::With => f.write_str("with"),
            Token::Where => f.write_str("where"),
            Token::Await => f.write_str("await"),
            Token::Struct => f.write_str("struct"),
            Token::Enum => f.write_str("enum"),
            Token::Union => f.write_str("union"),
            Token::Type => f.write_str("type"),
            Token::Effect => f.write_str("effect"),
            Token::Trait => f.write_str("trait"),
            Token::Handler => f.write_str("handler"),
            Token::Impl => f.write_str("impl"),
            Token::Identifier(ident) => f.write_str(ident),
            Token::IntLiteral(value) => f.write_str(format!("{}", value).as_str()),
            Token::FloatLiteral(value) => f.write_str(format!("{}", value).as_str()),
            Token::StringLiteral(value) => f.write_str(value.as_str()),
            Token::BoolLiteral(value) => f.write_str(format!("{}", value).as_str()),
            Token::Plus => f.write_str("+"),
            Token::Minus => f.write_str("-"),
            Token::Multiply => f.write_str("*"),
            Token::Divide => f.write_str("/"),
            Token::Assign => f.write_str("="),
            Token::Eq => f.write_str("=="),
            Token::Modulo => f.write_str("%"),
            Token::NotEq => f.write_str("!="),
            Token::Lt => f.write_str("<"),
            Token::LtEq => f.write_str("<="),
            Token::Gt => f.write_str(">"),
            Token::GtEq => f.write_str(">="),
            Token::And => f.write_str("&&"),
            Token::Or => f.write_str("||"),
            Token::Not => f.write_str("!"),
            Token::Pipe => f.write_str("|"),
            Token::Comma => f.write_str(","),
            Token::Dot => f.write_str("."),
            Token::QuestionMark => f.write_str("?"),
            Token::ExclamationPt => f.write_str("!"),
            Token::Colon => f.write_str(":"),
            Token::PathSep => f.write_str("::"),
            Token::Arrow => f.write_str("->"),
            Token::Semicolon => f.write_str(";"),
            Token::Newline => f.write_str("\n"),
            Token::Eof => f.write_str("EOF"),
            Token::Typeof => f.write_str("typeof"),
            Token::As => f.write_str("as"),
            Token::LParen => f.write_str("("),
            Token::RParen => f.write_str(")"),
            Token::LBrace => f.write_str("{"),
            Token::RBrace => f.write_str("}"),
            Token::LBracket => f.write_str("["),
            Token::RBracket => f.write_str("]"),
            Token::Range => f.write_str(".."),
            Token::Pipeline => f.write_str("|>"),
            Token::Increment => f.write_str("++"),
            Token::Decrement => f.write_str("--"),
        }
    }
}