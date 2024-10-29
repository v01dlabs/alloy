pub mod ty;

use core::fmt;

use thin_vec::ThinVec;

use self::ty::{Function, Ident, Pattern, RefKind, Ty, TypeOp};
use crate::{
    ast::ty::{GenericParam, Mutability},
    lexer::Token,
};

#[allow(non_snake_case)]
pub fn P<T: 'static>(value: T) -> Box<T> {
    Box::new(value)
}

/// Represents the precedence levels for operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd)]
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
    #[inline]
    pub fn from_token(token: &Token) -> Precedence {
        match token {
            Token::Eq | Token::NotEq => Precedence::Equality,
            Token::Lt | Token::LtEq | Token::Gt | Token::GtEq => Precedence::Comparison,
            Token::Plus | Token::Minus => Precedence::Term,
            Token::Multiply | Token::Divide | Token::Modulo => Precedence::Factor,
            Token::Not => Precedence::Unary,
            Token::And => Precedence::And,
            Token::Or => Precedence::Or,
            Token::Assign => Precedence::Assignment,
            Token::Pipeline => Precedence::Pipeline,
            Token::LParen => Precedence::Call,
            Token::Dot => Precedence::Call,
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
        attrs: ThinVec<FnAttr>,
        function: Function,
        body: ThinVec<Box<AstNode>>,
    },
    VariableDeclaration {
        name: Ident,
        attrs: ThinVec<BindAttr>,
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
        prev: Box<AstNode>,
        next: Box<AstNode>,
    },
    Identifier(Ident),
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),
    ArrayLiteral(ThinVec<Box<AstNode>>),
    EffectDeclaration {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        bounds: Option<TypeOp>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        members: ThinVec<Box<AstNode>>,
    },
    StructDeclaration {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        members: ThinVec<Box<AstNode>>,
    },
    EnumDeclaration {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        variants: ThinVec<Box<AstNode>>,
    },
    TraitDeclaration {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        bounds: Option<TypeOp>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        members: ThinVec<Box<AstNode>>,
    },
    UnionDeclaration {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        bounds: Option<TypeOp>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
    },
    ImplDeclaration {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        kind: ImplKind,
        target: Ident,
        target_generic_params: ThinVec<GenericParam>,
        bounds: Option<TypeOp>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        members: ThinVec<Box<AstNode>>,
    },
    WithClause(ThinVec<Box<WithClauseItem>>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum WhereClauseItem {
    Generic(GenericParam),
    Algebraic(TypeOp),
}

#[derive(Debug, Clone, PartialEq)]
pub enum WithClauseItem {
    Generic(GenericParam),
    Algebraic(TypeOp),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FnAttr {
    pub is_async: bool,
    pub is_shared: bool,
    pub effects: ThinVec<Box<WithClauseItem>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BindAttr {
    pub mutability: Mutability,
    pub ref_kind: Option<RefKind>,
}

impl BindAttr {
    pub fn new(is_mut: bool, ref_kind: Option<RefKind>) -> Self {
        BindAttr {
            mutability: if is_mut {
                Mutability::Mut
            } else {
                Mutability::Not
            },
            ref_kind,
        }
    }

    pub fn from_list(attrs: &[BindAttr]) -> Self {
        let mut mutability = Mutability::Not;
        let mut ref_kind = None;
        for attr in attrs {
            mutability = mutability.max(attr.mutability);
            ref_kind = ref_kind.or(attr.ref_kind);
        }
        BindAttr {
            mutability,
            ref_kind,
        }
    }
}

/// Represents binary operators in Alloy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
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

impl BinaryOperator {
    #[inline]
    pub fn from_token(token: &Token) -> Option<BinaryOperator> {
        match token {
            Token::Plus => Some(BinaryOperator::Add),
            Token::Minus => Some(BinaryOperator::Subtract),
            Token::Multiply => Some(BinaryOperator::Multiply),
            Token::Divide => Some(BinaryOperator::Divide),
            Token::Modulo => Some(BinaryOperator::Modulo),
            Token::Eq => Some(BinaryOperator::Equal),
            Token::NotEq => Some(BinaryOperator::NotEqual),
            Token::Lt => Some(BinaryOperator::LessThan),
            Token::Gt => Some(BinaryOperator::GreaterThan),
            Token::LtEq => Some(BinaryOperator::LessThanOrEqual),
            Token::GtEq => Some(BinaryOperator::GreaterThanOrEqual),
            Token::And => Some(BinaryOperator::And),
            Token::Or => Some(BinaryOperator::Or),
            Token::Assign => Some(BinaryOperator::Assign),
            Token::Pipeline => Some(BinaryOperator::Pipeline),
            _ => None,
        }
    }
}

/// Represents unary operators in Alloy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOperator {
    Negate,
    Not,
    Increment,
}

impl UnaryOperator {
    pub fn from_token(token: &Token) -> Option<UnaryOperator> {
        match token {
            Token::Not => Some(UnaryOperator::Not),
            Token::Minus => Some(UnaryOperator::Negate),
            Token::Increment => Some(UnaryOperator::Increment),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImplKind {
    Struct,
    Enum,
    Trait,
    Union,
    Handler,
    Infer,
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

#[derive(Debug, Clone, PartialEq)]
pub enum IntKind {
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    Int,
}

#[derive(Debug, Clone, PartialEq)]
pub enum UintKind {
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    Uint,
    Byte,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FloatKind {
    Float,
    F32,
    F64,
}

impl fmt::Display for IntKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntKind::Int => write!(f, "i64"),
            IntKind::I8 => write!(f, "i8"),
            IntKind::I16 => write!(f, "i16"),
            IntKind::I32 => write!(f, "i32"),
            IntKind::I64 => write!(f, "i64"),
            IntKind::I128 => write!(f, "i128"),
            IntKind::Isize => write!(f, "isize"),
        }
    }
}

impl fmt::Display for UintKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UintKind::Byte => write!(f, "u8"),
            UintKind::Uint => write!(f, "usize"),
            UintKind::U8 => write!(f, "u8"),
            UintKind::U16 => write!(f, "u16"),
            UintKind::U32 => write!(f, "u32"),
            UintKind::U64 => write!(f, "u64"),
            UintKind::U128 => write!(f, "u128"),
            UintKind::Usize => write!(f, "usize"),
        }
    }
}

impl fmt::Display for FloatKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FloatKind::Float => write!(f, "f64"),
            FloatKind::F32 => write!(f, "f32"),
            FloatKind::F64 => write!(f, "f64"),
        }
    }
}
