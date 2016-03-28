" for the plugin manager vundle
set nocompatible
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where vundle should install plugins
"call vundle#begin('~/some/path/here')

" let vundle manage vundle, required
Plugin 'gmarik/Vundle.vim' " changed from 'gmarik/vundle' which is differ from Vundle.vim
" plugin for verson control
Plugin 'mhinz/vim-signify'
" plugin for latex
Plugin 'latex-box-team/latex-box'
" for automatic complementation
" track the engine.
Plugin 'sirver/ultisnips'
" snippets are separated from the engine. add this if you want them:
Plugin 'honza/vim-snippets'
" fugitive
Plugin 'tpope/vim-fugitive'
Plugin 'sjl/gundo.vim'
Plugin 'kien/ctrlp.vim'
Plugin 'xolox/vim-notes'
Plugin 'scrooloose/nerdtree'
Plugin 'scrooloose/syntastic'
####################

" all of your plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required

" color scheme
syntax enable
set background=light
colorscheme solarized

" set line number
set number
" working directory is always same as the file being edited
set autochdir
" check spelling
set spell
" for the font sizes
set guifont=monaco:h12
""""""""""""""""""""""""""""""""""""""""
set cursorline          " highlight current line
filetype indent on      " load filetype-specific indent files
set wildmenu            " visual autocomplete for command menu
set lazyredraw          " redraw only when we need to.
set showmatch           " highlight matching [{()}]
set incsearch           " search as characters are entered
set hlsearch            " highlight matches
" fold
set foldenable          " enable folding
set foldlevelstart=10   " open most folds by default
set foldnestmax=10      " 10 nested fold max
set foldmethod=indent   " fold based on indent level
" move
" move to beginning/end of line
nnoremap b ^
nnoremap e $

" $/^ doesn't do anything
nnoremap $ <nop>
nnoremap ^ <nop>
" highlight last inserted text
nnoremap gv `[v`]
""""""""""""""""""""""""""""""""""""""""

nnoremap th  :tabfirst<cr>
nnoremap tj  :tabprev<cr>
nnoremap tk  :tabnext<cr>
nnoremap tl  :tablast<cr>
nnoremap td  :tabclose<cr>
nnoremap tn :tabnew<cr>

" yank & paste
set clipboard=unnamed

" set undo
" tell it to use an undo file
set undofile
" set a directory to store the undo history
set undodir=/home/yourname/.vimundo/

" display status line always
set laststatus=2

set statusline=   " clear the statusline for when vimrc is reloaded
set statusline+=%<%f                    " filename
set statusline+=\ %w%h%m%r               " options
set statusline+=%{fugitive#statusline()} " git hotness
set statusline+=\ [%{&ff}/%y]            " filetype
set statusline+=\ [%{getcwd()}]          " current dir
set statusline+=%=%-14.(%l,%c%v%)\ %p%%  " right aligned file nav info


if isdirectory(expand("~/.vim/bundle/vim-fugitive/"))
    nnoremap <silent> <leader>gs :gstatus<cr>
    nnoremap <silent> <leader>gd :gdiff<cr>
    nnoremap <silent> <leader>gc :gcommit<cr>
    nnoremap <silent> <leader>gb :gblame<cr>
    nnoremap <silent> <leader>gl :glog<cr>
    nnoremap <silent> <leader>gp :git push<cr>
    nnoremap <silent> <leader>gr :gread<cr>
    nnoremap <silent> <leader>gw :gwrite<cr>
    nnoremap <silent> <leader>ge :gedit<cr>
    " mnemonic _i_nteractive
    nnoremap <silent> <leader>gi :git add -p %<cr>
    nnoremap <silent> <leader>gg :signifytoggle<cr>
endif
" jk is escape
inoremap jk <esc>
" leader shortcuts
let mapleader=","       " leader is comma
" toggle gundo
nnoremap <leader>u :gundotoggle<cr>
" switch between splits
" nnoremap <c-j> <c-w><c-j>
" nnoremap <c-k> <c-w><c-k>
" nnoremap <c-l> <c-w><c-l>
" nnoremap <c-h> <c-w><c-h>
" open ctrlp in new tab
let g:ctrlp_prompt_mappings = {
    \ 'acceptselection("e")': ['<c-t>'],
    \ }
