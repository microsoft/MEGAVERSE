from typing import Any, Dict
import jinja2


def render_jinja_template(
    template: str,
    dataset_example: Dict[str, Any],
    **kwargs,
) -> str:
    """Renders jinja template with dataset_example

    Args:
        template (str): _description_
        dataset_example (Dict[str, Any]): _description_
        format_instructions (Dict[str, str]): _description_
        lang (str): _description_"""

    template = jinja2.Template(template)
    rendered_template = template.render(**dataset_example)
    return rendered_template
