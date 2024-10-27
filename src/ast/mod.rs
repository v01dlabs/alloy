pub mod ty;

use std::sync::Arc;

use thin_vec::ThinVec;

use crate::{ast::ty::{AttrItem, Const, GenericParam, Mutability, Path, QualifiedSelf}, error::TypeError, lexer::token::{self, Token}};
use self::ty::{Function, Ident, Pattern, RefKind, Ty, TypeOp};

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

#[derive(Debug, Clone, PartialEq)]
pub struct AstElem {
    pub id: usize,
    pub kind: AstElemKind,
    pub tokens: Arc<ThinVec<Token>>,
}

impl AstElem {
    pub fn item(item: Box<Item>) -> Self {
        let tokens = item.tokens.clone();
        AstElem {
            id: 0,
            kind: AstElemKind::Item(item),
            tokens
        }
    }
    pub fn expr(expr: Box<Expr>) -> Self {
        let tokens = expr.tokens.clone();
        AstElem {
            id: 0,
            kind: AstElemKind::Expr(expr),
            tokens
        }
    }
    pub fn statement(statement: Box<Statement>) -> Self {
        let tokens = statement.tokens.clone();
        AstElem {
            id: 0,
            kind: AstElemKind::Statement(statement),
            tokens,
        }
    }

    pub fn program(statements: ThinVec<Box<AstElem>>) -> Self {
        let tokens: ThinVec<_> = statements
            .iter()
            .map(|elem| elem.tokens.as_slice())
            .flatten()
            .map(|token| token.clone())
            .collect();
        AstElem {
            id: 0,
            kind: AstElemKind::Program(statements),
            tokens: Arc::new(tokens),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AstElemKind {
    Program(ThinVec<Box<AstElem>>),
    Expr(Box<Expr>),
    Item(Box<Item>),
    Statement(Box<Statement>),
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
pub enum Visibility {
    Local(Option<Path>),
    Public,
    Private,
    CrateLevel
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

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {    
    pub kind: ExprKind,
    pub tokens: Arc<ThinVec<Token>>,
}

impl Expr {

    pub fn call(callee: Box<Expr>, generic_args: Option<ThinVec<Box<Ty>>>, args: ThinVec<Box<Expr>>) -> Self {
        let tokens: ThinVec<Token> = callee.tokens.as_slice().iter()
            .chain(args.iter().map(|arg| arg.tokens.as_slice()).flatten()).cloned().collect();
        Self {
            kind: ExprKind::Call { callee, generic_args, args },
            tokens: Arc::new(tokens),
        }
    }

    pub fn path(path: Option<Box<QualifiedSelf>>, id: Path) -> Self {
        let tokens: ThinVec<Token> = path.as_ref().map(|path| path.ty.tokens.as_slice()).unwrap_or_default()
            .iter().cloned().collect();
        Self {
            kind: ExprKind::Path(path, id),
            tokens: Arc::new(tokens),
        }
    }

    pub fn field(expr: Box<Expr>, id: Ident) -> Self {
        let tokens: ThinVec<Token> = expr.tokens.as_slice().iter()
            .cloned().collect();
        Self {
            kind: ExprKind::Field(expr, id),
            tokens: Arc::new(tokens),
        }
    }

    pub fn method_call(path: String, receiver: Box<Expr>, args: ThinVec<Box<Expr>>) -> Self {
        let tokens: ThinVec<Token> = receiver.tokens.as_slice().iter()
            .chain(args.iter().map(|arg| arg.tokens.as_slice()).flatten()).cloned().collect();
        Self {
            kind: ExprKind::MethodCall { path_seg: path, receiver, args },
            tokens: Arc::new(tokens),
        }
    }
    pub fn unary(operator: UnaryOperator, operand: Box<Expr>) -> Self {
        let tokens = operand.tokens.clone();    
        Self {
            kind: ExprKind::Unary(operator, operand),
            tokens,
        }
    }

    pub fn binary(operator: BinaryOperator, left: Box<Expr>, right: Box<Expr>) -> Self {
        let tokens: ThinVec<Token> = left.tokens.as_slice().iter()
            .chain(right.tokens.as_slice()).cloned().collect();
        
        Self {
            kind: ExprKind::Binary{ binop: operator, lhs: left, rhs: right },
            tokens: Arc::new(tokens),
        }
    }

    pub fn cast(expr: Box<Expr>, ty: Box<Ty>) -> Self {
        let tokens: ThinVec<Token> = expr.tokens.as_slice().iter()
            .chain(ty.tokens.as_slice()).cloned().collect();
        Self {
            kind: ExprKind::Cast(expr, ty),
            tokens: Arc::new(tokens),
        }
    }
    pub fn literal(value: Literal) -> Self {
        Self { kind: ExprKind::Literal(value), tokens: Arc::new(ThinVec::new()) }
    }
    pub fn let_(pat: Box<Pattern>, ty: Box<Expr>, init: Box<Expr>) -> Self {
        let tokens: ThinVec<Token> = pat.tokens.as_slice().iter()
            .chain(ty.tokens.as_slice()).cloned().collect();
        Self {
            kind: ExprKind::Let { pat, ty, init },
            tokens: Arc::new(tokens),
        }
    }
    pub fn type_(expr: Box<Expr>, ty: Box<Ty>) -> Self {
        let tokens: ThinVec<Token> = expr.tokens.as_slice().iter()
            .chain(ty.tokens.as_slice()).cloned().collect();
        Self {
            kind: ExprKind::Type { expr, ty },
            tokens: Arc::new(tokens),
        }
    }
    
    pub fn if_(cond: Box<Expr>, then: Box<Expr>, else_: Option<Box<Expr>>) -> Self {
        let tokens: ThinVec<Token> = cond.tokens.as_slice().iter()
            .chain(then.tokens.as_slice()).cloned().collect();
        Self {
            kind: ExprKind::If { cond, then, else_ },
            tokens: Arc::new(tokens),
        }
    }
    
    pub fn while_(cond: Box<Expr>, body: Box<Expr>, label: Option<String>) -> Self {
        let tokens: ThinVec<Token> = cond.tokens.as_slice().iter()
            .chain(body.tokens.as_slice()).cloned().collect();
        Self {
            kind: ExprKind::While { cond, body, label },
            tokens: Arc::new(tokens),
        }
    }

    pub fn for_(pat: Box<Pattern>, iter: Box<Expr>, body: Box<Expr>, label: Option<String>) -> Self {
        let tokens: ThinVec<Token> = pat.tokens.as_slice().iter()
            .chain(iter.tokens.as_slice()).cloned().collect();
        Self {
            kind: ExprKind::For { pat, iter, body, label },
            tokens: Arc::new(tokens),
        }
    }

    pub fn loop_(body: Box<Expr>, label: Option<String>) -> Self {
        let tokens: ThinVec<Token> = body.tokens.as_slice().iter()
            .cloned().collect();
        Self {
            kind: ExprKind::Loop { body, label },
            tokens: Arc::new(tokens),
        }
    }

    pub fn match_(expr: Box<Expr>, arms: ThinVec<Arm>) -> Self {
        let tokens: ThinVec<Token> = expr.tokens.as_slice().iter()
            .chain(arms.iter().map(|arm| arm.tokens.as_slice()).flatten()).cloned().collect();
        Self {
            kind: ExprKind::Match { expr, arms },
            tokens: Arc::new(tokens),
        }
    }

    pub fn block(stmts: ThinVec<Box<Statement>>, label: Option<String>) -> Self {
        let tokens: ThinVec<Token> = stmts.iter()
            .map(|stmt| stmt.tokens.as_slice()).flatten().cloned().collect();
        let stmts = stmts.into_iter().map(|s| *s).collect();
        Self {
            kind: ExprKind::Block(P(Block { stmts }), label),
            tokens: Arc::new(tokens),
        }
    }
    pub fn assign(lhs: Box<Expr>, rhs: Box<Expr>) -> Self {
        let tokens: ThinVec<Token> = lhs.tokens.as_slice().iter()
            .chain(rhs.tokens.as_slice()).cloned().collect();
        Self {
            kind: ExprKind::Assign { lhs, rhs },
            tokens: Arc::new(tokens),
        }
    }
    
    pub fn assign_op(lhs: Box<Expr>, op: BinaryOperator, rhs: Box<Expr>) -> Self {
        let tokens: ThinVec<Token> = lhs.tokens.as_slice().iter()
            .chain(rhs.tokens.as_slice()).cloned().collect();
        Self {
            kind: ExprKind::AssignOp { lhs, op, rhs },
            tokens: Arc::new(tokens),
        }
    }

    pub fn closure(callee: Box<Expr>, params: ThinVec<Box<Pattern>>, closure: Box<Expr>) -> Self {
        let tokens: ThinVec<Token> = callee.tokens.as_slice().iter()
            .chain(params.iter().map(|param| param.tokens.as_slice()).flatten()).cloned().collect();
        Self {
            kind: ExprKind::Closure { callee, params, closure },
            tokens: Arc::new(tokens),
        }
    }

    pub fn trailing_closure(callee: Box<Expr>, args: ThinVec<Box<Expr>>, closure: Box<Expr>) -> Self {
        let tokens: ThinVec<Token> = callee.tokens.as_slice().iter()
            .chain(args.iter().map(|args| args.tokens.as_slice()).flatten()).cloned().collect();
        Self {
            kind: ExprKind::TrailingClosure { callee, args, closure },
            tokens: Arc::new(tokens),
        }
    }

    pub fn return_(value: Option<Box<Expr>>) -> Self {
        let tokens: ThinVec<Token> = if let Some(ref value) = value {
            value.tokens.iter().cloned().collect()
        } else { ThinVec::new() };
        Self {
            kind: ExprKind::Return(value),
            tokens: Arc::new(tokens),
        }
    }
    
    pub fn pipeline(prev: Box<AstElem>, next: Box<Expr>) -> Self {
        let tokens: ThinVec<Token> = prev.tokens.as_slice().iter()
            .chain(next.tokens.as_slice()).cloned().collect();
        Self {
            kind: ExprKind::PipelineOperation { prev, next },
            tokens: Arc::new(tokens),
        }
    }
    pub fn try_(expr: Box<Expr>) -> Self {
        let tokens: ThinVec<Token> = expr.tokens.as_slice().iter()
            .cloned().collect();
        Self {
            kind: ExprKind::Try(expr),
            tokens: Arc::new(tokens),
        }
    }
    pub fn bang(expr: Box<Expr>) -> Self {
        let tokens: ThinVec<Token> = expr.tokens.as_slice().iter()
            .cloned().collect();
        Self {
            kind: ExprKind::Unwrap(expr),
            tokens: Arc::new(tokens),
        }
    }

    pub fn run(expr: Box<Expr>) -> Self {
        let tokens: ThinVec<Token> = expr.tokens.as_slice().iter()
            .cloned().collect();
        Self {
            kind: ExprKind::Run(expr),
            tokens: Arc::new(tokens),
        }
    }

    pub fn array(elements: ThinVec<Box<Expr>>) -> Self {
        let tokens: ThinVec<Token> = elements.iter()
            .map(|elem| elem.tokens.as_slice()).flatten().cloned().collect();
        Self {
            kind: ExprKind::Array(elements),
            tokens: Arc::new(tokens),
        }
    }

    
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Array(ThinVec<Box<Expr>>),
    ConstBlock(Const),
    Call {
        callee: Box<Expr>,
        generic_args: Option<ThinVec<Box<Ty>>>,
        args: ThinVec<Box<Expr>>,
    },
    MethodCall {
        path_seg: String,
        receiver: Box<Expr>,
        args: ThinVec<Box<Expr>>,
    },
    Binary {
        binop: BinaryOperator,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Unary(UnaryOperator, Box<Expr>),
    Cast(Box<Expr>, Box<Ty>),
    Literal(Literal),
    Let {
        pat: Box<Pattern>,
        ty: Box<Expr>,
        init: Box<Expr>,
    },
    Type {
        expr: Box<Expr>,
        ty: Box<Ty>,
    },
    If {
        cond: Box<Expr>,
        then: Box<Expr>,
        else_: Option<Box<Expr>>,
    },
    While {
        cond: Box<Expr>,
        body: Box<Expr>,
        label: Option<String>,
    },
    For {
        pat: Box<Pattern>,
        iter: Box<Expr>,
        body: Box<Expr>,
        label: Option<String>,
    },
    Loop {
        body: Box<Expr>,
        label: Option<String>,
    },
    Match {
        expr: Box<Expr>,
        arms: ThinVec<Arm>,
    },
    Block(Box<Block>, Option<String>),
    Await(Box<Expr>),
    Assign {
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    AssignOp {
        lhs: Box<Expr>,
        op: BinaryOperator,
        rhs: Box<Expr>,
    },
    Closure {
        callee: Box<Expr>,
        params: ThinVec<Box<Pattern>>,
        closure: Box<Expr>,
    },
    TrailingClosure {
        callee: Box<Expr>,
        args: ThinVec<Box<Expr>>,
        closure: Box<Expr>,
    },
    Struct {
        qual_self: Option<Box<QualifiedSelf>>,
        path: Path,
        fields: ThinVec<Box<ExprField>>,
    },
    PipelineOperation {
        prev: Box<AstElem>,
        next: Box<Expr>,
    },
    Field(Box<Expr>, Ident),
    Index {
        expr: Box<Expr>,
        index: Box<Expr>,
    },
    Range {
        start: Box<Expr>,
        end: Box<Expr>,
        limits: RangeLimits,
    },
    Underscore,
    Paren(Box<Expr>),
    Path(Option<Box<QualifiedSelf>>, Path),
    Break {
        label: Option<String>,
        expr: Option<Box<Expr>>,
    },
    Continue { label: Option<String>},
    Return(Option<Box<Expr>>),
    Try(Box<Expr>), // expr?
    Unwrap(Box<Expr>), // expr!
    Run(Box<Expr>),
}


#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: ThinVec<Statement>,
}       

/// Local represents a `let` statement, e.g., `let <pat>:<ty> = <expr>;`.
#[derive(Clone, PartialEq, Debug)]
pub struct Local {
    pub pat: Box<Pattern>,
    pub ty: Option<Box<Ty>>,
    pub kind: LocalKind,
    pub attrs: ThinVec<AttrItem>,
    pub tokens: Arc<ThinVec<Token>>,
}

#[derive(Clone, PartialEq, Debug)]
pub enum LocalKind {
    /// Local declaration.
    /// Example: `let x;`
    Decl,
    /// Local declaration with an initializer.
    /// Example: `let x = y;`
    Init(Box<Expr>),
    /// Local declaration with an initializer and an `else` clause.
    /// Example: `let Some(x) = y else { return };`
    InitElse(Box<Expr>, Box<Block>),
}


impl LocalKind {
    pub fn init(&self) -> Option<&Expr> {
        match self {
            Self::Decl => None,
            Self::Init(i) | Self::InitElse(i, _) => Some(i),
        }
    }

    pub fn init_else_opt(&self) -> Option<(&Expr, Option<&Block>)> {
        match self {
            Self::Decl => None,
            Self::Init(init) => Some((init, None)),
            Self::InitElse(init, els) => Some((init, Some(els))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]  
pub struct Statement {
    pub kind: StatementKind,
    pub tokens: Arc<ThinVec<Token>>,
}

impl Statement {
    pub fn let_(pat: Box<Pattern>, ty: Option<Box<Ty>>, init: Option<Box<Expr>>) -> Self {
        let tokens: ThinVec<Token> = pat.tokens.as_slice().iter()
            .chain(ty.as_ref().map(|ty| ty.tokens.as_slice()).unwrap_or_default()).cloned().collect();
        let tokens =  Arc::new(tokens);
        let kind = if let Some(init) = init {
            LocalKind::Init(init)
        } else {
            LocalKind::Decl
        };
        Self {
            kind: StatementKind::Let(Box::new(Local { 
                pat, 
                ty, 
                kind, 
                attrs: ThinVec::new(),
                tokens: tokens.clone()
            })),
            tokens,
        }
    }

    pub fn binding(item: Box<Item>) -> Self {
        let tokens: ThinVec<Token> = item.tokens.as_slice().iter()
            .cloned().collect();
        Self {
            kind: StatementKind::Item(item),
            tokens: Arc::new(tokens),
        }
    }
    pub fn item(item: Box<Item>) -> Self {
        let tokens: ThinVec<Token> = item.tokens.as_slice().iter()
            .cloned().collect();
        Self {
            kind: StatementKind::Item(item),
            tokens: Arc::new(tokens),
        }
    }
    pub fn expr(expr: Box<Expr>) -> Self {
        let tokens: ThinVec<Token> = expr.tokens.as_slice().iter()
            .cloned().collect();
        Self {
            kind: StatementKind::Expr(expr),
            tokens: Arc::new(tokens),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StatementKind {
    Let(Box<Local>),
    Item(Box<Item>),
    Expr(Box<Expr>),
    Semicolon(Box<Expr>),
    Empty,
}


#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub attrs: ThinVec<AttrItem>,
    pub vis: Visibility,
    pub kind: ItemKind,
    pub tokens: Arc<ThinVec<Token>>,
}

impl Item {
    pub fn fn_(name: Ident, attrs: ThinVec<FnAttr>, function: Function, body: ThinVec<Box<Statement>>) -> Self {
        Item {
            attrs: ThinVec::new(),
            vis: Visibility::Local(None),
            kind: ItemKind::Fn {
                name,
                attrs,
                function,
                body,
            },
            tokens: Arc::new(ThinVec::new()),
        }
    }
    
    pub fn bind(name: Ident, attrs: ThinVec<BindAttr>, type_annotation: Option<Box<Ty>>, initializer: Option<Box<Expr>>) -> Self {
        Item {
            attrs: ThinVec::new(),
            vis: Visibility::Local(None),
            kind: ItemKind::Bind {
                name,
                attrs,
                type_annotation,
                initializer,
            },
            tokens: Arc::new(ThinVec::new()),
        }
    }

    pub fn effect(name: Ident, generic_params: ThinVec<GenericParam>, bounds: Option<TypeOp>, where_clause: ThinVec<Box<WhereClauseItem>>, members: ThinVec<Box<AstElem>>) -> Self {
        Item {
            attrs: ThinVec::new(),
            vis: Visibility::Local(None),
            kind: ItemKind::Effect {
                name,
                generic_params,
                bounds,
                where_clause,
                members,
            },
            tokens: Arc::new(ThinVec::new()),
        }
    }

    pub fn struct_(name: Ident, generic_params: ThinVec<GenericParam>, where_clause: ThinVec<Box<WhereClauseItem>>, members: ThinVec<Box<AstElem>>) -> Self {
        Item {
            attrs: ThinVec::new(),
            vis: Visibility::Local(None),
            kind: ItemKind::Struct {
                name,
                generic_params,
                where_clause,
                members,
            },
            tokens: Arc::new(ThinVec::new()),
        }
    }

    pub fn enum_(name: Ident, generic_params: ThinVec<GenericParam>, where_clause: ThinVec<Box<WhereClauseItem>>, variants: ThinVec<Box<AstElem>>) -> Self {
        Item {
            attrs: ThinVec::new(),
            vis: Visibility::Local(None),
            kind: ItemKind::Enum {
                name,
                generic_params,
                where_clause,
                variants,
            },
            tokens: Arc::new(ThinVec::new()),
        }
    }

    pub fn trait_(name: Ident, generic_params: ThinVec<GenericParam>, bounds: Option<TypeOp>, where_clause: ThinVec<Box<WhereClauseItem>>, members: ThinVec<Box<AstElem>>) -> Self {
        Item {
            attrs: ThinVec::new(),
            vis: Visibility::Local(None),
            kind: ItemKind::Trait {
                name,
                generic_params,
                bounds,
                where_clause,
                members,
            },
            tokens: Arc::new(ThinVec::new()),
        }
    }

    pub fn union_(name: Ident, generic_params: ThinVec<GenericParam>, bounds: Option<TypeOp>, where_clause: ThinVec<Box<WhereClauseItem>>) -> Self {
        Item {
            attrs: ThinVec::new(),
            vis: Visibility::Local(None),
            kind: ItemKind::Union {
                name,
                generic_params,
                bounds,
                where_clause,
            },
            tokens: Arc::new(ThinVec::new()),
        }
    }
    
    pub fn impl_(name: Ident, generic_params: ThinVec<GenericParam>, kind: ImplKind, target: Ident, target_generic_params: ThinVec<GenericParam>, bounds: Option<TypeOp>, where_clause: ThinVec<Box<WhereClauseItem>>, members: ThinVec<Box<AstElem>>) -> Self {
        Item {
            attrs: ThinVec::new(),
            vis: Visibility::Local(None),
            kind: ItemKind::Impl {
                name,
                generic_params,
                kind,
                target,
                target_generic_params,
                bounds,
                where_clause,
                members,
            },
            tokens: Arc::new(ThinVec::new()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ItemKind {
    Fn {
        name: Ident,
        attrs: ThinVec<FnAttr>,
        function: Function,
        body: ThinVec<Box<Statement>>,
    },
    Bind {
        name: Ident,
        attrs: ThinVec<BindAttr>,
        type_annotation: Option<Box<Ty>>,
        initializer: Option<Box<Expr>>,
    },
    Effect {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        bounds: Option<TypeOp>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        members: ThinVec<Box<AstElem>>, 
    },
    Struct {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        members: ThinVec<Box<AstElem>>, 
    },
    Enum {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        variants: ThinVec<Box<AstElem>>, 
    },
    Trait {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        bounds: Option<TypeOp>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        members: ThinVec<Box<AstElem>>, 
    },
    Union {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        bounds: Option<TypeOp>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
    },
    Impl {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        kind: ImplKind,
        target: Ident,
        target_generic_params: ThinVec<GenericParam>,
        bounds: Option<TypeOp>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        members: ThinVec<Box<AstElem>>, 
    },
}


#[derive(Debug, Clone, PartialEq)]
pub struct Arm {
    pub attrs: ThinVec<AttrItem>,
    pub pat: Box<Pattern>,
    pub guard: Option<Box<Expr>>,
    pub body: Option<Box<Expr>>,
    pub tokens: Arc<ThinVec<Token>>,
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

#[derive(Clone, Debug, PartialEq)]
pub struct ExprField {
    pub attrs: ThinVec<AttrItem>,
    pub ident: Ident,
    pub expr: Box<Expr>,
    pub is_shorthand: bool,
    pub is_placeholder: bool,
}

/// Limit types of a range (inclusive or exclusive).
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum RangeLimits {
    /// Inclusive at the beginning, exclusive at the end.
    HalfOpen,
    /// Inclusive at the beginning and end.
    Closed,
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


