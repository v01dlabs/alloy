use thin_vec::ThinVec;

use crate::ast::AstNode;

pub type Ident = String;

#[derive(Debug, Clone, PartialEq)]
pub enum IntTy {
    Int, // i64   
    Byte, // u8
    UInt, // usize
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ty {
    pub kind: TyKind,
}

impl Ty {
    pub fn simple(name: Ident) -> Self {
        Ty {
            kind: TyKind::Simple(name),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TyKind {
    Int(IntTy),
    Float, 
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
pub struct Const(pub Box<AstNode>);

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub generic_params: ThinVec<GenericParam>,
    pub inputs: ThinVec<Param>,
    pub output: FnRetTy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenericParam {
    pub name: Ident,
    pub kind: GenericParamKind,
    pub attrs: ThinVec<AttrItem>,
    pub bounds: Option<TypeOp>,
    pub is_placeholder: bool,
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
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenericParamKind {
    Type(Option<Box<Ty>>),
    Const {
        ty: Box<Ty>,
        value: Option<Const>
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: Ident,
    pub ty: Box<Ty>,
} 

#[derive(Debug, Clone, PartialEq)]
pub enum FnRetTy {
    Infer,
    Ty(Box<Ty>),  
}

impl Default for FnRetTy {
    fn default() -> Self {
        FnRetTy::Ty(Box::new(Ty {
            kind: TyKind::Tuple(ThinVec::new()),
        }))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttrItem {
    pub path: Path,
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


#[derive(Debug, Clone, PartialEq)]
pub struct Pattern {
    pub kind: PatternKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PatternKind {
    Wildcard,
    Ident(BindingMode, Ident, Option<Box<Pattern>>),
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

#[derive(Debug, Clone, PartialEq)]
pub struct Path {
    pub segments: ThinVec<Ident>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedSelf {
    pub ty: Box<Ty>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum RefKind {
    ThreadLocal(Mutability),
    Sync(Mutability),
}

