# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'SAIL'
copyright = '2021, Tucker Siegel'
author = 'Tucker Siegel'

import sail
from sail import modules
from sail import losses
from sail import random

print (losses)
print (dir(losses))

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 
              'sphinx.ext.autosummary', 
              'sphinx.ext.coverage', 
              'sphinx.ext.doctest', 
              "sphinx.ext.viewcode",
              'sphinx.ext.napoleon',
              'sphinx.ext.autosectionlabel',
              ]

autosummary_generate = True
numpydoc_show_class_members = False

autosectionlabel_prefix_document = True


from sphinx.writers import html, html5

def replace(Klass):
    old_call = Klass.visit_reference

    def visit_reference(self, node):
        if 'refuri' in node and 'generated' in node.get('refuri'):
            ref = node.get('refuri')
            ref_anchor = ref.split('#')
            if len(ref_anchor) > 1:
                # Only add the id if the node href and the text match,
                # i.e. the href is "torch.flip#torch.flip" and the content is
                # "torch.flip" or "flip" since that is a signal the node refers
                # to autogenerated content
                anchor = ref_anchor[1]
                txt = node.parent.astext()
                if txt == anchor or txt == anchor.split('.')[-1]:
                    self.body.append('<p id="{}"/>'.format(ref_anchor[1]))
        return old_call(self, node)
    Klass.visit_reference = visit_reference

replace(html.HTMLTranslator)
replace(html5.HTML5Translator)

from sphinx.util.docfields import TypedField
from sphinx import addnodes
import sphinx.ext.doctest

# Without this, doctest adds any example with a `>>>` as a test
doctest_test_doctest_blocks = ''
doctest_default_flags = sphinx.ext.doctest.doctest.ELLIPSIS
doctest_global_setup = ''''''

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "sail-logo.png"
html_favicon = 'sail-favicon.ico'


html_theme_options = {
  "navigation_depth": 1
}



# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']