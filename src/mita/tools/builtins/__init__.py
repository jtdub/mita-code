"""Built-in tool definitions and handlers."""

from __future__ import annotations

from mita.tools.builtins.file_edit import TOOL_DEF as FILE_EDIT_DEF
from mita.tools.builtins.file_edit import execute as file_edit_execute
from mita.tools.builtins.file_read import TOOL_DEF as FILE_READ_DEF
from mita.tools.builtins.file_read import execute as file_read_execute
from mita.tools.builtins.file_write import TOOL_DEF as FILE_WRITE_DEF
from mita.tools.builtins.file_write import execute as file_write_execute
from mita.tools.builtins.git import TOOL_DEF as GIT_DEF
from mita.tools.builtins.git import execute as git_execute
from mita.tools.builtins.glob_tool import TOOL_DEF as GLOB_DEF
from mita.tools.builtins.glob_tool import execute as glob_execute
from mita.tools.builtins.grep_tool import TOOL_DEF as GREP_DEF
from mita.tools.builtins.grep_tool import execute as grep_execute
from mita.tools.builtins.shell import TOOL_DEF as SHELL_DEF
from mita.tools.builtins.shell import execute as shell_execute

BUILTIN_TOOLS: dict[str, tuple[object, object]] = {
    FILE_READ_DEF.name: (FILE_READ_DEF, file_read_execute),
    FILE_WRITE_DEF.name: (FILE_WRITE_DEF, file_write_execute),
    FILE_EDIT_DEF.name: (FILE_EDIT_DEF, file_edit_execute),
    GLOB_DEF.name: (GLOB_DEF, glob_execute),
    GREP_DEF.name: (GREP_DEF, grep_execute),
    SHELL_DEF.name: (SHELL_DEF, shell_execute),
    GIT_DEF.name: (GIT_DEF, git_execute),
}
