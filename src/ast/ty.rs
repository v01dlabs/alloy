use std::{fmt, sync::Arc};

use thin_vec::{thin_vec, ThinVec};

use crate::{
    ast::{
        AstElem, BindAttr,
    }, 
    lexer::token::Token, 
    type_checker::Type}
;


pub type Ident = String;

#[derive(Debug, Clone, PartialEq)]
pub struct Ty {
    pub kind: TyKind,
    pub tokens: Arc<ThinVec<Token>>,
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            TyKind::Int(ty) => write!(f, "{ty}"),
            TyKind::Uint(ty) => write!(f, "{ty}"),
            TyKind::Float(ty) => write!(f, "{ty}"),
            TyKind::Bool => write!(f, "bool"),
            TyKind::Char => write!(f, "char"),
            TyKind::String => write!(f, "string"),
            TyKind::Array(ty) => write!(f, "[{ty}]"),
            TyKind::Path(path) => write!(f, "{path}"),
            TyKind::Tuple(types) => {
                write!(f, "(")?;
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty}")?;
                }
                write!(f, ")")
            }
            TyKind::SizedArray(ty, size) => write!(f, "[{ty}; {size}]"),
            TyKind::Function(function) => write!(f, "{function}"),
            TyKind::Const(const_) => write!(f, "{const_}"),
            TyKind::Algebraic(type_op) => write!(f, "{type_op}"),
            TyKind::Generic(name, types) => {
                write!(f, "{name}")?;
                write!(f, "<")?;
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty}")?;
                }
                write!(f, ">")
            }
            TyKind::Paren(ty) => write!(f, "({ty})"),
            TyKind::Ref(kind, ty) => write!(f, "{kind} {ty}"),
            TyKind::Pattern(ty, pat) => {
                write!(f, "{ty}")?;
                write!(f, " {pat}")
            }
            TyKind::Simple(name) => write!(f, "{name}"),
            TyKind::Any => write!(f, "any"),
            TyKind::Infer => write!(f, "?"),
            TyKind::SelfType => write!(f, "self"),
            TyKind::Never => write!(f, "never"),
            TyKind::Err => write!(f, "err"),
        }
    }
}

impl Ty {
    pub fn simple(name: Ident) -> Self {
        Ty {
            kind: TyKind::Simple(name),
            tokens: Arc::new(ThinVec::new()),
        }
    }

    pub fn fn_(inputs: ThinVec<Param>, output: FnRetTy, generic_params: ThinVec<GenericParam>) -> Self {
        Ty {
            kind: TyKind::Function(Function {
                generic_params,
                inputs,
                output,
            }),
            tokens: Arc::new(ThinVec::new()),
        }
    }

    pub fn generic(name: Ident, params: ThinVec<Box<Ty>>) -> Self {
        Ty {
            kind: TyKind::Generic(name, params),
            tokens: Arc::new(ThinVec::new()),
        }
    }

    pub fn infer() -> Self {
        Ty {
            kind: TyKind::Infer,
            tokens: Arc::new(ThinVec::new()),
        }
    }

    pub fn self_type() -> Self {        
        Ty {
            kind: TyKind::SelfType,
            tokens: Arc::new(ThinVec::new()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TyKind {
    Int(IntKind),
    Uint(UintKind),
    Float(FloatKind),
    Bool,
    Char,
    String,
    Array(Box<Ty>),
    Path(Path),
    Tuple(ThinVec<Box<Ty>>),
    SizedArray(Box<Ty>, Const),
    Function(Function),
    Const(Const),
    Algebraic(TypeOp),
    Generic(Ident, ThinVec<Box<Ty>>),
    Paren(Box<Ty>),
    Ref(RefKind, Box<Ty>),
    Pattern(Box<Ty>, Box<Pattern>),
    Simple(Ident),
    Any,
    Infer,
    SelfType,
    Never,
    Err,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Const(pub Box<AstElem>);

impl fmt::Display for Const {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}   

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub generic_params: ThinVec<GenericParam>,
    pub inputs: ThinVec<Param>,
    pub output: FnRetTy,
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fn(")?;
        for (i, param) in self.inputs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{param}")?;
        }
        write!(f, ") -> {}", self.output)   
    }
    
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenericParam {
    pub name: Ident,
    pub kind: GenericParamKind,
    pub attrs: ThinVec<AttrItem>,
    pub bounds: Option<TypeOp>,
    pub is_placeholder: bool,
}

impl fmt::Display for GenericParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl GenericParam {
    pub fn simple(name: Ident) -> Self {
        GenericParam {
            name,
            kind: GenericParamKind::Type(None),
            attrs: ThinVec::new(),
            bounds: None,
            is_placeholder: false,
        }
    }

    pub fn const_(name: Ident, ty: Box<Ty>, value: Option<Const>) -> Self {
        GenericParam {
            name,
            kind: GenericParamKind::Const { ty, value },
            attrs: ThinVec::new(),
            bounds: None,
            is_placeholder: false,
        }
    }

    pub fn type_(name: Ident, ty: Box<Ty>) -> Self {
        GenericParam {
            name,
            kind: GenericParamKind::Type(Some(ty)),
            attrs: ThinVec::new(),
            bounds: None,
            is_placeholder: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenericParamKind {
    Type(Option<Box<Ty>>),
    Const { ty: Box<Ty>, value: Option<Const> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub ty: Box<Ty>,
}

impl fmt::Display for Param {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.ty)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FnRetTy {
    Infer,
    Ty(Box<Ty>),
}

impl fmt::Display for FnRetTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FnRetTy::Infer => write!(f, "_"),
            FnRetTy::Ty(ty) => write!(f, "{ty}"),
        }
    }
}   

impl FnRetTy {
    pub fn to_type(&self) -> Option<Type> {
        match self {
            FnRetTy::Infer => None,
            FnRetTy::Ty(ty) => Some(Type::from(*ty.clone())),
        }
    }
}

impl Default for FnRetTy {
    fn default() -> Self {
        FnRetTy::Ty(Box::new(Ty {
            kind: TyKind::Tuple(ThinVec::new()),
            tokens: Arc::new(ThinVec::new()),
        }))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttrItem {
    pub path: Path,
}

impl fmt::Display for AttrItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.path)
    }
}   

#[derive(Debug, Clone, PartialEq)]
pub enum TypeOp {
    And(ThinVec<Box<Ty>>),
    Or(ThinVec<Box<Ty>>),
    Xor(ThinVec<Box<Ty>>),
    Not(Box<Ty>),
    Subset(Box<Ty>),
    Implements(Box<Ty>),
}

impl fmt::Display for TypeOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TypeOp::And(thin_vec) => {
                write!(f, "( ")?;
                for (i, ty) in thin_vec.iter().enumerate() {
                    if i > 0 {
                        write!(f, " + ")?;
                    }
                    write!(f, "{ty}")?;
                }
                write!(f, " ) ")
            }
            TypeOp::Or(thin_vec) => {
                write!(f, "( ")?;
                for (i, ty) in thin_vec.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty}")?;
                }
                write!(f, " ) ")
            }
            TypeOp::Xor(thin_vec) => {
                write!(f, "( ")?;
                for (i, ty) in thin_vec.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{ty}")?;
                }
                write!(f, " ) ")
            }
            TypeOp::Not(ty) => write!(f, "!{ty} "),
            TypeOp::Subset(ty) => write!(f, ": {ty} "),
            TypeOp::Implements(ty) => write!(f, "impl {ty} "),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub kind: PatternKind,
    pub tokens: Arc<ThinVec<Token>>,
}

impl Pattern {
    pub fn ident(mode: BindAttr, ident: Ident, pat: Option<Box<Pattern>>) -> Self {
        Pattern {
            kind: PatternKind::Ident(mode, ident, pat),
            tokens: Arc::new(ThinVec::new()),
        }
    }
    pub fn id_simple(ident: Ident) -> Self {
        Pattern {
            kind: PatternKind::Ident(BindAttr::new(false, None), ident, None),
            tokens: Arc::new(ThinVec::new()),
        }
    }   
}

impl Pattern {
    pub fn to_simple(&self) -> Option<Ident> {
        match self.kind {
            PatternKind::Ident(_, ref ident, _) => Some(ident.clone()),
            _ => None,
        }
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            PatternKind::Wildcard => write!(f, "_"),
            PatternKind::Ident(bind, ident, pat) => {
                if let Some(pat) = pat {
                    write!(f, "{bind}{ident}: {pat}")
                } else {
                    write!(f, "{bind}{ident}")
                }
            }
            PatternKind::Tuple(patterns) => {
                write!(f, "(")?;
                for (i, pattern) in patterns.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{pattern}")?;
                }
                write!(f, ")")
            }
            PatternKind::TupleStruct(qual_self, path, patterns) => {
                if let Some(qual_self) = qual_self {
                    write!(f, "{qual_self}")?;
                }
                write!(f, "{path}")?;
                write!(f, "(")?;
                for (i, pattern) in patterns.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{pattern}")?;
                }
                write!(f, ")")
            }
            PatternKind::Struct(qual_self, path, patterns) => {
                if let Some(qual_self) = qual_self {
                    write!(f, "{qual_self}")?;
                }
                write!(f, "{path}")?;
                write!(f, "{{")?;
                for (i, pat_field) in patterns.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{pat_field}")?;
                }
                write!(f, "}}")
            }
            PatternKind::Path(qual_self, path) => {
                if let Some(qual_self) = qual_self {
                    write!(f, "{qual_self}")?;
                }
                write!(f, "{path}")
            }
            PatternKind::Ref(pat, kind) => {
                write!(f, "{kind} {pat}")
            }
            PatternKind::Or(thin_vec) => {
                write!(f, "(")?;
                for (i, pattern) in thin_vec.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{pattern}")?;
                }
                write!(f, ")")
            },
            PatternKind::TypeOp(type_op, pattern) => {
                write!(f, "{type_op} {pattern}")
            }
            PatternKind::Paren(pattern) => todo!(),
            PatternKind::Never => todo!(),
            PatternKind::Err => todo!(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PatternKind {
    Wildcard,
    Ident(BindAttr, Ident, Option<Box<Pattern>>),
    Tuple(ThinVec<Box<Pattern>>),
    TupleStruct(Option<Box<QualifiedSelf>>, Path, ThinVec<Box<Pattern>>),
    Struct(Option<Box<QualifiedSelf>>, Path, ThinVec<PatField>),
    Path(Option<Box<QualifiedSelf>>, Path),
    Ref(Box<Pattern>, RefKind),
    Or(ThinVec<Box<Pattern>>),
    TypeOp(TypeOp, Box<Pattern>),
    Paren(Box<Pattern>),
    Never,
    Err,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PatField {
    /// The identifier for the field.
    pub ident: Ident,
    /// The pattern the field is destructured to.
    pub pat: Box<Pattern>,
    pub is_shorthand: bool,
    pub attrs: ThinVec<AttrItem>,
    pub is_placeholder: bool,
}

impl fmt::Display for PatField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} #[{:?}]", self.ident, self.pat, self.attrs)
    }
}

/// The mode of a binding (`mut`, `ref mut`, etc).
/// Used for both the explicit binding annotations given in the HIR for a binding
/// and the final binding mode that we infer after type inference/match ergonomics.
/// `.0` is the by-reference mode (`ref`, `ref mut`, or by value),
/// `.1` is the mutability of the binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindingMode(pub ByRef, pub Mutability);

impl BindingMode {
    pub const NONE: Self = Self(ByRef::No, Mutability::Not);
    pub const REF: Self = Self(ByRef::Yes(Mutability::Not), Mutability::Not);
    pub const MUT: Self = Self(ByRef::No, Mutability::Mut);
    pub const REF_MUT: Self = Self(ByRef::Yes(Mutability::Mut), Mutability::Not);
    pub const MUT_REF: Self = Self(ByRef::Yes(Mutability::Not), Mutability::Mut);
    pub const MUT_REF_MUT: Self = Self(ByRef::Yes(Mutability::Mut), Mutability::Mut);

    pub fn prefix_str(self) -> &'static str {
        match self {
            Self::NONE => "",
            Self::REF => "ref ",
            Self::MUT => "mut ",
            Self::REF_MUT => "ref mut ",
            Self::MUT_REF => "mut ref ",
            Self::MUT_REF_MUT => "mut ref mut ",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ByRef {
    Yes(Mutability),
    No,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Copy)]
pub enum Mutability {
    // N.B. Order is deliberate, so that Not < Mut
    Not,
    Mut,
}

impl fmt::Display for Mutability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Mutability::Not => write!(f, ""),
            Mutability::Mut => write!(f, "mut "),
        }
    }
    
}

#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    pub segments: ThinVec<Ident>,
}

impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.segments.join("::"))
    }
}

impl Path {
    pub fn new(segments: ThinVec<Ident>) -> Self {
        Self { segments }
    }

    pub fn ident(ident: Ident) -> Self {
        Path {
            segments: thin_vec![ident],
        }
    }

    pub fn concat(a: Path, b: Path) -> Path {
        Path::new(a.segments.into_iter().chain(b.segments.into_iter()).collect())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedSelf {
    pub ty: Box<Ty>,
}

impl fmt::Display for QualifiedSelf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ty)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RefKind {
    ThreadLocal(Mutability),
    Sync(Mutability),
}

impl fmt::Display for RefKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RefKind::ThreadLocal(mutability) => write!(f, "ref {mutability}"),
            RefKind::Sync(mutability) => write!(f, "shared {mutability}"),
        }
    }
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
            IntKind::Int => write!(f, "i32"),
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