if exists("b:current_syntax")
  finish
endif

syn match ffComment ";.*$"
syn match ffSuperComment ";;.*$"
syn match ffMeta    "^#\w\+"
syn match ffMacro   "\$\w\+" 

syn match ffMetaError "^#meta\s\+\([^{]\+\)$"

"syn region ffSection start='\[' end='\]'  contains=ffSectionName,ffSectionSpecial
syn match ffSectionName   "\[\s*\([A-Za-z0-9_-]\|\s\|\.\)\+\s*\]"
syn match ffSectionRemove   "\[\s*\(![A-Za-z0-9_-]\+\)\+\s*\]"
syn match ffSectionSpecial   "\[\s*\(edges\|non-edges\|patterns\|molmeta\|features\)\s*\]"

syn match ffMolType   "^\s*\[\s*\(moleculetype\)\s*\]"
syn match ffLink      "^\s*\[\s*\(link\)\s*\]"
syn match ffVariables      "^\s*\[\s*\(variables\)\s*\]"
syn match ffMacroSectionName     "^\s*\[\s*\(macros\)\s*\]"

syn region ffMacroSection start='^\s*\[\s*macros\s*\]' end='^\s*\['me=e-1 transparent contains=ALL keepend
syn match ffMacroDef  "^\(\w\+\)" contained containedin=ffMacroSection

syn region ffMolecules start='^\s*\[\s*moleculetype\s*\]' end='^\s*\['me=e-1 transparent contains=ALL keepend
syn match ffBlockNameError "^\([A-Za-z0-9_-]\|+\)\+" contained containedin=ffMolecules
syn match ffBlockName "^\([A-Za-z0-9_-]\|+\)\+" contained containedin=ffMolecules nextgroup=ffMolNumber skipwhite
syn match ffMolNumber "\d\+\s*$" contained containedin=ffMolecules


hi def link ffComment      Comment
hi def link ffSuperComment VimCommentTitle
hi def link ffMeta         PreProc
hi def link ffMacro        Constant
hi def link ffSectionName   Identifier
hi def link ffSectionRemove Structure
hi def link ffSectionSpecial    Type
hi def link ffMolNumber    level14c
hi def link ffMolType      Exception
hi def link ffLink         Exception
hi def link ffVariables    Exception
hi def link ffMacroSectionName Exception
hi def link ffMacroDef     Constant
hi def link ffBlockName    String	
hi def link ffMetaError    Error


let b:current_syntax = "ff"
