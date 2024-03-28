# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'python-language.so',

  # Include one or more languages
  [
    'tree-sitter-python'
  ]
)

