" for the plugin manager vundle
set fileencodings=utf-8,gb2312,gb18030,gbk,ucs-bom,cp936,latin1
set nocompatible
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where vundle should install plugins
"call vundle#begin('~/some/path/here')

" let vundle manage vundle, required
Plugin 'VundleVim/Vundle.vim' " changed from 'gmarik/vundle' which is differ from Vundle.vim
Plugin 'Valloric/YouCompleteMe'
" plugin for version control
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
Bundle 'xolox/vim-misc'
Bundle 'xolox/vim-notes'
Plugin 'scrooloose/nerdtree'
Plugin 'scrooloose/syntastic'
" ####################

" all of your plugins must be added before the following line
call vundle#end()            " required
" Automatic indent
filetype plugin indent on    " required
" show existing tab with 4 spaces width
set tabstop=4
" when indenting with '>', use 4 spaces width
set shiftwidth=4
" On pressing tab, insert 4 spaces
set expandtab

filetype plugin on    " required
syntax enable           " enable syntax processing
let g:solarized_termcolors=256
set background=dark
colorscheme solarized

set tags+=/Users/wujie/SourceCode/tags
set path+=/Users/wujie/SourceCode/ShearBands/**
set number              " show line numbers
" working directory is always same as the file being edited
set autochdir
" check spelling
"set spell
" for the font sizes
set guifont=Monaco:h12
""""""""""""""""""""""""""""""""""""""""
set cursorline          " highlight current line
set wildmenu            " visual autocomplete for command menu
set lazyredraw          " redraw only when we need to.
set showmatch           " highlight matching [{()}]
set incsearch           " search as characters are entered
set hlsearch            " highlight matches
" fold
set foldenable          " enable folding
set foldlevelstart=10   " open most folds by default
set foldnestmax=10      " 10 nested fold max
nnoremap <space> za     " space open/closes folds
set foldmethod=marker   " fold based on marker 
set foldcolumn=4
" move
" move to beginning/end of line
nnoremap B ^
nnoremap E $
" $/^ doesn't do anything
nnoremap $ <nop>
nnoremap ^ <nop>
" move vertically by visual line
nnoremap j gj
nnoremap k gk

" ctags
map <C-\> :tab split<CR>:exec("tag ".expand("<cword>"))<CR>

" highlight last inserted text
nnoremap gV `[v`]
""""""""""""""""""""""""""""""""""""""""
" leader shortcuts
let mapleader=","       " leader is comma
" jk is escape
inoremap jk <esc>
" CtrlP settings
let g:ctrlp_match_window = 'bottom,order:ttb'
let g:ctrlp_switch_buffer = 0
let g:ctrlp_working_path_mode = 0
" let g:ctrlp_user_command = 'ag %s -l --nocolor --hidden -g ""'
let g:ycm_global_ycm_extra_conf = '~/.vim/bundle/YouCompleteMe/third_party/ycmd/cpp/ycm/.ycm_extra_conf.py'

nnoremap th  :tabfirst<cr>
nnoremap J   :tabprev<cr>
nnoremap K   :tabnext<cr>
nnoremap tl  :tablast<cr>
nnoremap td  :tabclose<cr>
nnoremap tn :tabnew<cr>

" yank & paste
set clipboard=unnamed

" set undo
" tell it to use an undo file
set undofile
" set a directory to store the undo history
set undodir=/Users/wujie/.vimundo/

" display status line always
set laststatus=2

set statusline=   " clear the statusline for when vimrc is reloaded
set statusline+=%<%f                    " filename
set statusline+=\ %w%h%m%r               " options
set statusline+=%{fugitive#statusline()} " git hotness
set statusline+=\ [%{&ff}/%y]            " filetype
set statusline+=\ [%{getcwd()}]          " current dir
set statusline+=%=%-14.(%l,%c%v%)\ %p%%  " right aligned file nav info

" if isdirectory(expand("~/.vim/bundle/vim-fugitive/"))
nnoremap <leader>gs :Gstatus<cr>
nnoremap <leader>gd :Gdiff<cr>
nnoremap <leader>gc :Gcommit<cr>
nnoremap <leader>gb :Gblame<cr>
nnoremap <leader>gl :Glog<cr>
nnoremap <leader>gp :Git push<cr>
nnoremap <leader>gr :Gread<cr>
nnoremap <leader>gw :Gwrite<cr>
nnoremap <leader>ge :Gedit<cr>
" mnemonic _i_nteractive
nnoremap <leader>gi :Git add -p %<cr>
nnoremap <leader>gg :SignifyToggle<cr>
" endif
" toggle gundo
nnoremap <leader>u :GundoToggle<CR>
" switch between splits
" nnoremap <c-j> <c-w><c-j>
" nnoremap <c-k> <c-w><c-k>
" nnoremap <c-l> <c-w><c-l>
" nnoremap <c-h> <c-w><c-h>
" open ctrlp in new tab
let g:ctrlp_prompt_mappings = {
    \ 'acceptselection("e")': ['<c-t>'],
    \ }
" complete
let g:UltiSnipsExpandTrigger="<tab>"
let g:UltiSnipsJumpForwardTrigger="<c-b>"
let g:UltiSnipsJumpBackwardTrigger="<c-z>"
