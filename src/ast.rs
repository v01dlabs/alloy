
use thin_vec::ThinVec;

use crate::lexer::Token;
use crate::ty::{Const, Function, Ident, Ty};


/// Represents the precedence levels for operators.
#[derive(Debug, PartialEq, Eq, PartialOrd)]
pub enum Precedence {
    None,
    Assignment, // =
    Pipeline,   // |>
    Or,         // ||
    And,        // &&
    Equality,   // == !=
    Comparison, // < > <= >=
    Term,       // + -
    Factor,     // / * 
    Unary,      // - !
    Call,       // . ()
    Primary,
}

impl Precedence {
    fn from_token(token: &Token) -> Precedence {
        match token {
            Token::Eq | Token::NotEq => Precedence::Equality,
            Token::Lt | Token::LtEq | Token::Gt | Token::GtEq => Precedence::Comparison,
            Token::Plus | Token::Minus => Precedence::Term,
            Token::Multiply | Token::Divide => Precedence::Factor,
            Token::Not => Precedence::Unary,
            Token::And => Precedence::And,
            Token::Or => Precedence::Or,
            Token::Assign => Precedence::Assignment,
            Token::Pipeline => Precedence::Pipeline,
            Token::LParen => Precedence::Call,
            _ => Precedence::None,
        }
    }
}


/// Represents a node in the Abstract Syntax Tree (AST).
#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {
    Program(ThinVec<Box<AstNode>>),
    FunctionDeclaration {
        name: Ident,
        function: Function,
        body: ThinVec<Box<AstNode>>,
    },
    VariableDeclaration {
        name: Ident,
        mutable: bool,
        type_annotation: Option<Box<Ty>>,
        initializer: Option<Box<AstNode>>,
    },
    IfStatement {
        condition: Box<AstNode>,
        then_branch: Box<AstNode>,
        else_branch: Option<Box<AstNode>>,
    },
    WhileLoop {
        condition: Box<AstNode>,
        body: Box<AstNode>,
    },
    ForInLoop {
        item: Ident,
        iterable: Box<AstNode>,
        body: Box<AstNode>,
    },
    GuardStatement {
        condition: Box<AstNode>,
        body: Box<AstNode>,
    },
    ReturnStatement(Option<Box<AstNode>>),
    Block(ThinVec<Box<AstNode>>),
    BinaryOperation {
        left: Box<AstNode>,
        operator: BinaryOperator,
        right: Box<AstNode>,
    },
    UnaryOperation {
        operator: UnaryOperator,
        operand: Box<AstNode>,
    },
    FunctionCall {
        callee: Box<AstNode>,
        arguments: ThinVec<Box<AstNode>>,
    },
    GenericFunctionCall {
        name: Ident,
        generic_args: ThinVec<Box<Ty>>,
        arguments: ThinVec<Box<AstNode>>,
    },
    TrailingClosure {
        callee: Box<AstNode>,
        closure: Box<AstNode>,
    },
    PipelineOperation {
        left: Box<AstNode>,
        right: Box<AstNode>,
    },
    Identifier(Ident),
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),
    ArrayLiteral(ThinVec<Box<AstNode>>),
}

/// Represents binary operators in Alloy.
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    And,
    Or,
    Assign,
    Pipeline,
}

/// Represents unary operators in Alloy.
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Negate,
    Not,
    Increment,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Byte(u8),
    UInt(usize),
    Float(f64),
    String(String),
    Bool(bool),
    Char(char),
}
