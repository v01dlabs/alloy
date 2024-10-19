use thin_vec::ThinVec;

use crate::ast::{AstNode, P};


#[derive(Debug, Clone, PartialEq)]
pub enum IntTy {
    Int(i64),   
    Byte(u8),
    UInt(usize),
}



#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    Int(IntTy),
    Float(f64), 
    Bool(bool),
    Char(char), 
    String(String),
    Array(P<Ty>),
    Tuple(ThinVec<P<Ty>>),
    SizedArray(P<Ty>, Const),
    Function(Function),
    Const(Const),
    TypeOp(TypeOp),
    Generic(String, ThinVec<Ty>),
    Paren(P<Ty>),
    Any,
    Infer,
    SelfType,
    Never,
}


#[derive(Debug, Clone, PartialEq)]
pub struct Const(pub P<AstNode>);

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub generic_params: ThinVec<GenericParam>,
    pub inputs: ThinVec<Param>,
    pub output: FnRetTy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenericParam {
    pub name: String,
    pub kind: GenericParamKind,
    pub attrs: ThinVec<AttrItem>,
    pub bounds: TypeOp,
    pub is_placeholder: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenericParamKind {
    Type(Option<P<Ty>>),
    Const {
        ty: P<Ty>,
        value: Option<Const>
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {} 

#[derive(Debug, Clone, PartialEq)]
pub enum FnRetTy {
    Infer(String),
    Ty(P<Ty>),  
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttrItem {
    pub path: String,
}

#[derive(Debug, Clone, PartialEq)] 
pub enum TypeOp {
    And(ThinVec<P<Ty>>),
    Or(ThinVec<P<Ty>>),
    Xor(ThinVec<P<Ty>>),
    Not(P<Ty>),
    Subset(P<Ty>),
    Implements(P<Ty>),
}
