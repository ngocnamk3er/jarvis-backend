"""One message for "could not read that file from the sandbox", shared by the
tools that open a sandbox file (view_image, check_pdf, read_pdf, office_*).

A 404 from the sandbox is the usual case and has one fix: the file is not
there yet. The agent hits it by calling a tool in the same step as the
workspace_copy_to_sandbox that is meant to put the file there, and the raw error ("Failed to
communicate with the sandbox at http://...") told it nothing, so it gave up on
the tool. Say what is wrong and what to do.
"""


def read_error(name: str, e: Exception) -> str:
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status == 404 or isinstance(e, FileNotFoundError):
        return (f"Error: {name} is not in the sandbox. If it is in the user's workspace, copy it in "
                f"with workspace_copy_to_sandbox first and wait for that to finish before using it.")
    return f"Error: could not read {name} from the sandbox ({type(e).__name__}: {e})."
