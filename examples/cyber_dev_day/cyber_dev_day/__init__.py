import logging
from pprint import PrettyPrinter, pprint
import sys
from textwrap import dedent
import time

# class NotebookStreamHandler(logging.StreamHandler):

#     def __init__(self, stream=None):
#         super().__init__(stream=stream)

#         self._last_check_time = -1
#         self._printer: PrettyPrinter | None = None

#         try:
#             from IPython.core.getipython import get_ipython
#             from IPython.display import display, HTML

#             self._display = display
#             self._html = HTML

#             if get_ipython() is not None:
#                 self._is_notebook = True
#         except ImportError:
#             self._display = None
#             self._html = None

#     def _get_printer(self) -> PrettyPrinter:

#         assert self._is_notebook, "Inconsistent state: _get_printer called when not in a notebook."

#         if (time.time() - self._last_check_time > 5):
#             # Get the width of the notebook
#             js = """<script>
#             IPython.notebook.kernel.execute("cell_width="+($( ".cell").width()))
#             </script>"""
#             self._display(self._html(js), display_id="cell_width", update=True)

#             time.sleep(3)

#             js2 = dedent("""
#             <canvas id="canvas"></canvas>
#             <script>
#                 // retrieve the width and font
#                 var el = document.querySelector("div.CodeMirror-lines")
#                 var ff = window.getComputedStyle(el, null).getPropertyValue('font');
#                 var widthpxl = el.clientWidth

#                 //set up canvas to measure text width
#                 var can = document.getElementById('canvas');
#                 var ctx = can.getContext('2d');
#                 ctx.font = ff;

#                 //measure one char of text and compute num char in one line
#                 var txt = ctx.measureText('A');
#                 alert(Math.floor(widthpxl/txt.width))
#                 //EDIT: to populate python variable with the output:
#                 IPython.notebook.kernel.execute("ncols=" + Math.floor(widthpxl/txt.width));
#             </script>
#             """).strip("\n")

#             self._printer = PrettyPrinter(width=eval("cell_width") - 2)

#             self._last_check_time = time.time()

#         return self._printer

#     def emit(self, record):

#         try:
#             msg = self.format(record)
#             stream = self.stream

#             if (self._is_notebook):
#                 printer = self._get_printer()

#                 msg = printer.pformat(msg)

#             # issue 35046: merged two stream.writes into one.
#             stream.write(msg + self.terminator)
#             self.flush()
#         except RecursionError:  # See issue 36272
#             raise
#         except Exception:
#             self.handleError(record)

#         if self._is_notebook:
#             try:
#                 from IPython.display import display, HTML
#                 display(HTML(self.format(record)))
#             except ImportError:
#                 super().emit(record)
#         else:
#             super().emit(record)

morpheus_logger = logging.getLogger("morpheus")

if (not getattr(morpheus_logger, "_configured_by_morpheus", False)):

    # Set the morpheus logger to propagate upstream
    morpheus_logger.propagate = False

    # Add a default handler to the morpheus logger to print to screen
    morpheus_logger.addHandler(logging.StreamHandler(stream=sys.stdout))

    # Set a flag to indicate that the logger has been configured by Morpheus
    setattr(morpheus_logger, "_configured_by_morpheus", True)

logger = logging.getLogger(__name__)

# Set the parent logger for the entire package to use morpheus so we can take advantage of configure_logging
logger.parent = morpheus_logger
