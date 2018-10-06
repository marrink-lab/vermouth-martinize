VIM syntax highlighting for vermouth file formats
=================================================

This directory contains syntax highlighting rules for VIM to handle vermouth
files. Install them by copying the content of this directory in
`~/.vim/syntax/`. To activate the highlighting, you should copy the following
lines in `~/.vimrc`:

```
" Highlighting for vermouth force field files
au BufNewFile,BufRead *.ff set syntax=ff
```
