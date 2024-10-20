//! Type checker for Alloy
//!
//! This module is responsible for performing semantic analysis and type checking
//! on the Abstract Syntax Tree (AST) produced by the parser. It ensures that
//! the program is well-typed according to Alloy's type system.

use thin_vec::ThinVec;

use crate::{
    ast::{AstNode, BinaryOperator, UnaryOperator}, 
    ty::{AttrItem, BindingMode, Const, Function, Ident, IntTy, PatField, Path, Pattern, PatternKind, RefKind, Ty, TyKind, TypeOp}
};
use std::collections::HashMap;



/// Represents a typing error.
#[derive(Debug)]
pub struct TypeError {
    pub message: String,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Type Error: {}", self.message)
    }
}

/// The type environment stores variable and function types.
type TypeEnv = HashMap<String, Box<Type>>;

/// The main type checker struct.
pub struct TypeChecker {
    env: TypeEnv,
}

impl TypeChecker {
    /// Creates a new TypeChecker instance.
    pub fn new() -> Self {
        TypeChecker {
            env: HashMap::new(),
        }
    }

    pub fn infer_type(&mut self, ast: &AstNode) -> Result<Type, TypeError> {
        todo!()
    }


    pub fn typecheck_array_literal(&mut self, elements: &ThinVec<Box<AstNode>>) -> Result<Type, TypeError> {
        if elements.is_empty() {
            return Ok(Type::Array(Box::new(Type::Infer)));
        }
        let mut first_type = self.infer_type(&elements.first().unwrap())?;
        for element in elements.iter().skip(1) {
            let element_type = self.infer_type(&element)?;
            if first_type == Type::Infer || first_type == Type::Any {
                // If we couldn't immediately figure out the first type, maybe the next one will work
                first_type = element_type.clone();
            }
            if first_type != element_type {
                return Err(TypeError {
                    message: format!(
                        "Inconsistent types in array literal: {:?} and {:?}",
                        first_type, element_type
                    ),
                });
            }
        }

        Ok(Type::Array(Box::new(first_type)))
    }
}

/// Type checks an AST node.
pub fn typecheck(ast: &AstNode) -> Result<(), TypeError> {
    let mut checker = TypeChecker::new();
    //checker.typecheck_program(ast)?;
    Ok(())
}


/// Represents a type in the Alloy type system.
#[derive(Debug, PartialEq, Clone)]
pub enum Type {
    Int(IntTy),
    Float, 
    Bool,
    Char, 
    String,
    Array(Box<Type>),
    Path(Path),
    Tuple(ThinVec<Box<Type>>),
    SizedArray(Box<Type>, Const),
    Function(Function),
    Const(Const),
    Algebraic(AlgebraicType),
    Generic(Ident, ThinVec<Box<Type>>),
    Paren(Box<Type>),
    Ref(RefKind, Box<Type>),
    Pattern(Box<Type>, Box<PatternType>), 
    Simple(Ident),
    Any,
    Infer,
    SelfType,
    Never,
    Err,
}

impl From<Ty> for Type {
    fn from(ty: Ty) -> Self {
        match ty.kind {
            TyKind::Int(ty) => Type::Int(ty),
            TyKind::Float => Type::Float,
            TyKind::Bool => Type::Bool,
            TyKind::Char => Type::Char,
            TyKind::String => Type::String,
            TyKind::Array(ty) => Type::Array(Box::new(Type::from(*ty))),
            TyKind::Path(path) => Type::Path(path),
            TyKind::Tuple(types) => Type::Tuple(types.into_iter().map(|ty| Box::new(Type::from(*ty))).collect()),   
            TyKind::SizedArray(ty, size) => Type::SizedArray(Box::new(Type::from(*ty)), size),
            TyKind::Function(function) => Type::Function(function),
            TyKind::Const(const_) => Type::Const(const_),   
            TyKind::Algebraic(type_op) => Type::Algebraic(type_op.into()),   
            TyKind::Generic(name, types) => Type::Generic(name, types.into_iter().map(|ty| Box::new(Type::from(*ty))).collect()),
            TyKind::Paren(ty) => Type::Paren(Box::new(Type::from(*ty))),
            TyKind::Ref(kind, ty) => Type::Ref(kind, Box::new(Type::from(*ty))),
            TyKind::Pattern(ty, pat) => Type::Pattern(Box::new(Type::from(*ty)), Box::new(PatternType::from(*pat))),
            TyKind::Simple(name) => Type::Simple(name),
            TyKind::Any => Type::Any,
            TyKind::Infer => Type::Infer,
            TyKind::SelfType => Type::SelfType,
            TyKind::Never => Type::Never,
            TyKind::Err => Type::Err,
        }
    }
}

#[derive(Debug, Clone, PartialEq)] 
pub enum AlgebraicType {
    And(ThinVec<Box<Type>>),
    Or(ThinVec<Box<Type>>),
    Xor(ThinVec<Box<Type>>),
    Not(Box<Type>),
    Subset(Box<Type>),
    Implements(Box<Type>),
}

impl From<TypeOp> for AlgebraicType {
    fn from(type_op: TypeOp) -> Self {
        match type_op {
            TypeOp::And(types) => AlgebraicType::And(types.into_iter().map(|ty| Box::new(Type::from(*ty))).collect()),
            TypeOp::Or(types) => AlgebraicType::Or(types.into_iter().map(|ty| Box::new(Type::from(*ty))).collect()),
            TypeOp::Xor(types) => AlgebraicType::Xor(types.into_iter().map(|ty| Box::new(Type::from(*ty))).collect()),
            TypeOp::Not(ty) => AlgebraicType::Not(Box::new(Type::from(*ty))),
            TypeOp::Subset(ty) => AlgebraicType::Subset(Box::new(Type::from(*ty))),
            TypeOp::Implements(ty) => AlgebraicType::Implements(Box::new(Type::from(*ty))),
        }
    }
}


#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    Wildcard,
    Ident(BindingMode, Ident, Option<Box<PatternType>>),
    Tuple(ThinVec<Box<PatternType>>),
    TupleStruct(Option<Box<QualifiedSelf>>, Path, ThinVec<Box<PatternType>>),
    Struct(Option<Box<QualifiedSelf>>, Path, ThinVec<PatternField>),
    Path(Option<Box<QualifiedSelf>>, Path),
    Ref(Box<PatternType>, RefKind),
    Or(ThinVec<Box<PatternType>>),
    Algebraic(AlgebraicType, Box<PatternType>),
    Paren(Box<PatternType>),
    Never,
    Err,
}


impl From<Pattern> for PatternType {
    fn from(pattern: Pattern) -> Self {
        match pattern.kind {
            PatternKind::Wildcard => PatternType::Wildcard,
            PatternKind::Ident(mode, name, pat) => PatternType::Ident(
                mode, name, pat.map(|pat| Box::new(PatternType::from(*pat)))
            ),
            PatternKind::Tuple(patterns) => PatternType::Tuple(
                patterns.into_iter().map(|pat| Box::new(PatternType::from(*pat))).collect()
            ),
            PatternKind::TupleStruct(qual_self, path, patterns) => PatternType::TupleStruct(
                qual_self.map(|qual_self| Box::new(QualifiedSelf::from(*qual_self))), 
                path, patterns.into_iter().map(|pat| Box::new(PatternType::from(*pat))).collect()
            ),
            PatternKind::Struct(qual_self, path, patterns) => PatternType::Struct(
                qual_self.map(|qual_self| Box::new(QualifiedSelf::from(*qual_self))), 
                path, patterns.into_iter().map(|pat| PatternField::from(pat)).collect()
            ),
            PatternKind::Path(qual_self, path) => PatternType::Path(
                qual_self.map(|qual_self| Box::new(QualifiedSelf::from(*qual_self))), path
            ),
            PatternKind::Ref(pat, kind) => PatternType::Ref(Box::new(PatternType::from(*pat)), kind),
            PatternKind::Or(patterns) => PatternType::Or(
                patterns.into_iter().map(|pat| Box::new(PatternType::from(*pat))).collect()
            ),
            PatternKind::TypeOp(type_op, pat) => PatternType::Algebraic(
                type_op.into(), Box::new(PatternType::from(*pat))
            ),
            PatternKind::Paren(pat) => PatternType::Paren(Box::new(PatternType::from(*pat))),
            PatternKind::Never => PatternType::Never,
            PatternKind::Err => PatternType::Err,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedSelf {
    pub ty: Box<Type>,
}

impl From<crate::ty::QualifiedSelf> for QualifiedSelf {
    fn from(qual_self: crate::ty::QualifiedSelf) -> Self {
        QualifiedSelf {
            ty: Box::new(Type::from(*qual_self.ty)),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PatternField {
    /// The identifier for the field.
    pub ident: Ident,
    /// The pattern the field is destructured to.
    pub pat: Box<PatternType>,
    pub is_shorthand: bool,
    pub attrs: ThinVec<AttrItem>,
    pub is_placeholder: bool,
}

impl From<PatField> for PatternField {
    fn from(pat_field: PatField) -> Self {
        PatternField {
            ident: pat_field.ident,
            pat: Box::new(PatternType::from(*pat_field.pat)),
            is_shorthand: pat_field.is_shorthand,
            attrs: pat_field.attrs,
            is_placeholder: pat_field.is_placeholder,
        }
    }
}
