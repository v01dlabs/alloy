pub mod ty;

use std::{fmt, sync::Arc};

use thin_vec::ThinVec;

use crate::ast::ty::{
        AttrItem, Const, GenericParam, Mutability, Path, QualifiedSelf
    };
use self::ty::{Function, Ident, Pattern, RefKind, Ty, TypeOp};
use crate::lexer::token::Token;

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

impl fmt::Display for AstElem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            AstElemKind::Program(thin_vec) => {
                write!(f, "(")?;
                for (i, ast_elem) in thin_vec.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ast_elem}")?;
                }
                write!(f, ")")
            }
            AstElemKind::Expr(expr) => write!(f, "{expr}"),
            AstElemKind::Item(item) => write!(f, "{item}"),
            AstElemKind::Statement(statement) => write!(f, "{statement}"),
        }
    }
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
    Program(ThinVec<Box<AstElem>>),
    FunctionDeclaration {
        name: Ident,
        attrs: ThinVec<FnAttr>,
        function: Function,
        body: ThinVec<Box<AstElem>>,
    },
    VariableDeclaration {
        name: Ident,
        attrs: ThinVec<BindAttr>,
        type_annotation: Option<Box<Ty>>,
        initializer: Option<Box<AstElem>>,
    },
    IfStatement {
        condition: Box<AstElem>,
        then_branch: Box<AstElem>,
        else_branch: Option<Box<AstElem>>,
    },
    WhileLoop {
        condition: Box<AstElem>,
        body: Box<AstElem>,
    },
    ForInLoop {
        item: Ident,
        iterable: Box<AstElem>,
        body: Box<AstElem>,
    },
    GuardStatement {
        condition: Box<AstElem>,
        body: Box<AstElem>,
    },
    ReturnStatement(Option<Box<AstElem>>),
    Block(ThinVec<Box<AstElem>>),
    BinaryOperation {
        left: Box<AstElem>,
        operator: BinaryOperator,
        right: Box<AstElem>,
    },
    UnaryOperation {
        operator: UnaryOperator,
        operand: Box<AstElem>,
    },
    FunctionCall {
        callee: Box<AstElem>,
        arguments: ThinVec<Box<AstElem>>,
    },
    GenericFunctionCall {
        name: Ident,
        generic_args: ThinVec<Box<Ty>>,
        arguments: ThinVec<Box<AstElem>>,
    },
    TrailingClosure {
        callee: Box<AstElem>,
        closure: Box<AstElem>,
    },
    PipelineOperation {
        prev: Box<AstElem>,
        next: Box<AstElem>,
    },
    Identifier(Ident),
    IntLiteral(i64),
    FloatLiteral(f64),
    StringLiteral(String),
    BoolLiteral(bool),
    ArrayLiteral(ThinVec<Box<AstElem>>),
    EffectDeclaration {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        bounds: Option<TypeOp>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        members: ThinVec<Box<AstElem>>,
    },
    StructDeclaration {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        members: ThinVec<Box<AstElem>>,
    },
    EnumDeclaration {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        variants: ThinVec<Box<AstElem>>,
    },
    TraitDeclaration {
        name: Ident,
        generic_params: ThinVec<GenericParam>,
        bounds: Option<TypeOp>,
        where_clause: ThinVec<Box<WhereClauseItem>>,
        members: ThinVec<Box<AstElem>>,
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
        members: ThinVec<Box<AstElem>>,
    },
    WithClause(ThinVec<Box<WithClauseItem>>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum WhereClauseItem {
    Generic(GenericParam),
    Algebraic(TypeOp),
}

impl fmt::Display for WhereClauseItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WhereClauseItem::Generic(generic_param) => write!(f, "{}", generic_param),
            WhereClauseItem::Algebraic(type_op) => write!(f, "{}", type_op),
        }
    }
}   


#[derive(Debug, Clone, PartialEq)]
pub enum WithClauseItem {
    Generic(GenericParam),
    Algebraic(TypeOp),
}

impl fmt::Display for WithClauseItem {        
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {        
        match self {
            WithClauseItem::Generic(generic_param) => write!(f, "{}", generic_param),
            WithClauseItem::Algebraic(type_op) => write!(f, "{}", type_op),
        }
    }
}   

#[derive(Debug, Clone, PartialEq)]
pub struct FnAttr {
    pub is_async: bool,
    pub is_shared: bool,
    pub effects: ThinVec<Box<WithClauseItem>>,
}

impl fmt::Display for FnAttr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_vec(f, &self.effects)
    }
    
}

impl FnAttr {
    pub fn with_clause(effects: ThinVec<Box<WithClauseItem>>) -> Self {
        Self {
            is_async: false,
            is_shared: false,
            effects,
        }
    }
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

impl fmt::Display for BindAttr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.ref_kind {
            None => match self.mutability {
                Mutability::Not => write!(f, ""),
                Mutability::Mut => write!(f, "mut "),
        
            },
            Some(RefKind::ThreadLocal(Mutability::Not)) => write!(f, "ref "),
            Some(RefKind::ThreadLocal(Mutability::Mut)) => write!(f, "mut ref "),
            Some(RefKind::Sync(Mutability::Not)) => write!(f, "shared "),
            Some(RefKind::Sync(Mutability::Mut)) => write!(f, "mut shared "),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {    
    pub kind: ExprKind,
    pub tokens: Arc<ThinVec<Token>>,
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ExprKind::Array(thin_vec) => {
                write!(f, "[")?;
                for (i, expr) in thin_vec.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{expr}")?;
                }
                write!(f, "]")
            }
            ExprKind::ConstBlock(const_) => write!(f, "{const_}"),
            ExprKind::Call { callee, generic_args, args } => {
                write!(f, "{callee}")?;
                if let Some(generic_args) = generic_args {
                    write!(f, "[")?;
                    for(i, ty) in generic_args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{ty}")?;
                    }
                    write!(f, "]")?;
                }
                write!(f, "(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            ExprKind::MethodCall { path_seg, receiver, args } => {
                write!(f, "{receiver}.{path_seg}")?;
                write!(f, "(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
            ExprKind::Binary { binop, lhs, rhs } => {
                write!(f, "( {lhs} {binop} {rhs} )")
            }   
            ExprKind::Unary(unary_operator, expr) => {
                match unary_operator {
                    UnaryOperator::Negate => write!(f, "-{expr}"),
                    UnaryOperator::Not => write!(f, "!{expr}"),
                    UnaryOperator::Increment => write!(f, "{expr}++"),
                    
                }
            }
            ExprKind::Cast(expr, ty) => {
                write!(f, "{expr} as {ty}")
            }
            ExprKind::Literal(literal) => write!(f, "{literal}"),
            ExprKind::Let { pat, ty, init } => {
                write!(f, "let {pat}")?;
                if let Some(ty) = ty {
                    write!(f, ": {ty}")?;
                }
                if let Some(init) = init {
                    write!(f, " = {init}")?;
                }
                Ok(())
            }
            ExprKind::Type { expr, ty } => write!(f, "({expr}): {ty}"),
            ExprKind::Guard { condition, body } => write!(f, "guard {condition} {body}"),
            ExprKind::If { cond, then, else_ } => {
                write!(f, "if {cond} {then}")?;
                if let Some(else_) = else_ {
                    write!(f, " else {else_}")?;
                }
                Ok(())
            }
            ExprKind::While { cond, body, label } => {
                write!(f, "while {cond} {body}")?;
                if let Some(label) = label {
                    write!(f, " {label}")?;
                }
                Ok(())
            }
            ExprKind::For { pat, iter, body, label } => {
                write!(f, "for {pat} in {iter} {body}")?;   
                if let Some(label) = label {
                    write!(f, " {label}")?;
                }   
                Ok(())
            }
            ExprKind::Loop { body, label } => {
                write!(f, "loop {body}")?;
                if let Some(label) = label {
                    write!(f, " {label}")?;
                }
                Ok(())
            }
            ExprKind::Match { expr, arms } => todo!(),
            ExprKind::Block(block, _) => write!(f, "{block}"),
            ExprKind::Await(expr) => todo!(),
            ExprKind::Assign { lhs, rhs } => todo!(),
            ExprKind::AssignOp { lhs, op, rhs } => todo!(),
            ExprKind::Closure { callee, params, closure } => todo!(),
            ExprKind::TrailingClosure { callee, args, closure } => todo!(),
            ExprKind::Struct { qual_self, path, fields } => todo!(),
            ExprKind::PipelineOperation { prev, next } => todo!(),
            ExprKind::Field(expr, _) => todo!(),
            ExprKind::Index { expr, index } => todo!(),
            ExprKind::Range { start, end, limits } => todo!(),
            ExprKind::Underscore => todo!(),
            ExprKind::Paren(expr) => todo!(),
            ExprKind::Path(qualified_self, path) => todo!(),
            ExprKind::Break { label, expr } => todo!(),
            ExprKind::Continue { label } => todo!(),
            ExprKind::Return(expr) => todo!(),
            ExprKind::Try(expr) => todo!(),
            ExprKind::Unwrap(expr) => todo!(),
            ExprKind::Run(expr) => todo!(),
        }
    }
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

    pub fn guard(condition: Box<Expr>, body: Box<Expr>) -> Self {
        let tokens: ThinVec<Token> = condition.tokens.as_slice().iter()
            .chain(body.tokens.as_slice()).cloned().collect();
        Self {
            kind: ExprKind::Guard { condition, body },
            tokens: Arc::new(tokens),
        }
    }

    pub fn literal(value: Literal) -> Self {
        Self { kind: ExprKind::Literal(value), tokens: Arc::new(ThinVec::new()) }
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
    pub fn let_(pat: Box<Pattern>, ty: Option<Box<Expr>>, init: Option<Box<Expr>>) -> Self {
        let tokens: ThinVec<Token> = pat.tokens.as_slice().iter()
            .chain(ty.as_ref().map(|ty| ty.tokens.as_slice()).unwrap_or_default()).cloned().collect();
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
        ty: Option<Box<Expr>>,
        init: Option<Box<Expr>>,
    },
    Type {
        expr: Box<Expr>,
        ty: Box<Ty>,
    },
    Guard {
        condition: Box<Expr>,
        body: Box<Expr>,
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

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;   
        for (i, statement) in self.stmts.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{statement}")?;
        }
        write!(f, "}}")
    }
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

impl fmt::Display for Local {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "let ")?;
        write_vec(f, &self.attrs)?;
        write!(f, "{} ", self.pat)?;
        if let Some(ty) = &self.ty {
            write!(f, ": {}", ty)?;
        }
        if let LocalKind::Init(init) = &self.kind {
            write!(f, " = {}", init)?;
        }
        Ok(())
    }
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

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            StatementKind::Let(local) => write!(f, "{local}"),
            StatementKind::Item(item) => write!(f, "{item}"),
            StatementKind::Expr(expr) => write!(f, "{expr}"),
            StatementKind::Semicolon(expr) => write!(f, "{expr};"),
            StatementKind::Empty => write!(f, ";"),
        }
    }
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

fn write_vec<T: fmt::Display>(f: &mut fmt::Formatter<'_>, vec: &ThinVec<T>) -> fmt::Result {
    let mut first = true;
    for elem in vec.iter() {
        if first {
            first = false;
        } else {
            write!(f, ", ")?;
        }
        write!(f, "{elem}")?;
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub attrs: ThinVec<AttrItem>,
    pub vis: Visibility,
    pub kind: ItemKind,
    pub tokens: Arc<ThinVec<Token>>,
}

impl fmt::Display for Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ItemKind::Fn { name, attrs, function, body } => {
                write!(f, "fn {}", name)?;
                write!(f, "(")?;
                write_vec(f, &function.inputs)?;
                write!(f, ") -> {}", function.output)?;
                if attrs.is_empty() {
                    write!(f, "with ")?;
                    write_vec(f, attrs)?;
                    write!(f, "\n")?;
                }
                write!(f, "\n{{")?;
                write_vec(f, body)?;
                write!(f, "\n}}\n")
            }
            ItemKind::Bind { name, attrs, type_annotation, initializer } => {
                write!(f, "let ")?;
                write_vec(f, attrs)?;
                write!(f, " {}", name)?; 
                if let Some(type_annotation) = type_annotation {
                    write!(f, ": {}", type_annotation)?;
                }
                if let Some(initializer) = initializer {
                    write!(f, " = {}\n", initializer)?;
                }
                Ok(())
            }
            ItemKind::Effect { name, generic_params, bounds, where_clause, members } => {
                write!(f, "effect {}", name)?;
                write!(f, "[")?;
                write_vec(f, generic_params)?;
                write!(f, "]")?;
                if let Some(bounds) = bounds {
                    write!(f,": {}", bounds)?;
                }
                if !where_clause.is_empty() {
                    write!(f, "where\n")?;
                    write_vec(f, where_clause)?;
                    write!(f, "\n")?;
                }
                write!(f, "{{\n")?;
                write_vec(f, members)?;
                write!(f, "\n}}\n")
            }
            ItemKind::Struct { name, generic_params, where_clause, members } => {
                write!(f, "struct {}", name)?;
                write!(f, "[")?;
                write_vec(f, generic_params)?;
                write!(f, "]")?;
                if !where_clause.is_empty() {
                    write!(f, "where\n")?;
                    write_vec(f, where_clause)?;
                    write!(f, "\n")?;
                }
                write!(f, "{{\n")?;
                write_vec(f, members)?;
                write!(f, "\n}}\n")
            }
            ItemKind::Enum { name, generic_params, where_clause, variants } => {
                write!(f, "enum {}", name)?;
                write!(f, "[")?;
                write_vec(f, generic_params)?;
                write!(f, "]")?;
                if !where_clause.is_empty() {
                    write!(f, "where\n")?;
                    write_vec(f, where_clause)?;
                    write!(f, "\n")?;
                }
                write!(f, "{{\n")?;
                write_vec(f, variants)?;
                write!(f, "\n}}\n")
            }
            ItemKind::Trait { name, generic_params, bounds, where_clause, members } => {                
                write!(f, "trait {}", name)?;
                write!(f, "[")?;
                write_vec(f, generic_params)?;
                write!(f, "]")?;
                if let Some(bounds) = bounds {
                    write!(f, ": {}", bounds)?;
                }
                if !where_clause.is_empty() {
                    write!(f, "where\n")?;
                    write_vec(f, where_clause)?;
                    write!(f, "\n")?;
                }                
                write!(f, "{{\n")?;                
                write_vec(f, members)?;
                write!(f, "\n}}\n")
            }
            ItemKind::Union { name, generic_params, bounds, where_clause } => {
                write!(f, "union {}", name)?;
                write!(f, "[")?;
                write_vec(f, generic_params)?;
                write!(f, "]")?;
                if let Some(bounds) = bounds {
                    write!(f, ": {}", bounds)?;
                }
                if !where_clause.is_empty() {
                    write!(f, "where\n")?;
                    write_vec(f, where_clause)?;
                    write!(f, "\n")?;
                }
                Ok(())
            }
            ItemKind::Impl { name, generic_params, kind, target, target_generic_params, bounds, where_clause, members } => {
                write!(f, "impl {}", name)?;
                write!(f, "[")?;
                write_vec(f, generic_params)?;
                write!(f, "]")?;
                if let ImplKind::Struct = kind {
                    write!(f, "for ")?;
                } else {
                    write!(f, "for ")?;
                    write_vec(f, target_generic_params)?;
                    write!(f, ": ")?;
                    write!(f, "{}", target)?;
                }
                if let Some(bounds) = bounds {
                    write!(f, ": {}", bounds)?;
                }
                if !where_clause.is_empty() {
                    write!(f, "where\n")?;
                    write_vec(f, where_clause)?;
                    write!(f, "\n")?;
                }
                write!(f, "{{\n")?;
                write_vec(f, members)?;
                write!(f, "\n}}\n")
            }   
        }
    }
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

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOperator::Add => write!(f, "+"),
            BinaryOperator::Subtract => write!(f, "-"),
            BinaryOperator::Multiply => write!(f, "*"),
            BinaryOperator::Divide => write!(f, "/"),
            BinaryOperator::Modulo => write!(f, "%"),
            BinaryOperator::Equal => write!(f, "=="),
            BinaryOperator::NotEqual => write!(f, "!="),
            BinaryOperator::LessThan => write!(f, "<"),
            BinaryOperator::GreaterThan => write!(f, ">"),
            BinaryOperator::LessThanOrEqual => write!(f, "<="),
            BinaryOperator::GreaterThanOrEqual => write!(f, ">="),
            BinaryOperator::And => write!(f, "&&"),
            BinaryOperator::Or => write!(f, "||"),
            BinaryOperator::Assign => write!(f, "="),
            BinaryOperator::Pipeline => write!(f, "|>"),
        }
    }
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

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Int(value) => write!(f, "{value}"),
            Literal::Byte(value) => write!(f, "{value}"),
            Literal::UInt(value) => write!(f, "{value}"),
            Literal::Float(value) => write!(f, "{value}"),
            Literal::String(value) => write!(f, "\"{value}\""),
            Literal::Bool(value) => write!(f, "{value}"),
            Literal::Char(value) => write!(f, "'{value}'"),
        }
    }
}

