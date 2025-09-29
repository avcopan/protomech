"""Read MESS file format."""

from pathlib import Path

import automol
import pyvis

from .surf import Surface


def display(
    surf: Surface,
    height: str = "750px",
    out_name: str = "net.html",
    out_dir: str | Path = ".pyvis",
    stereo: bool = True,
    open_browser: bool = True,
) -> None:
    """Display surface as a pyvis Network.

    :param surf: Surface
    :param height: Frame height
    """
    out_dir = out_dir if isinstance(out_dir, Path) else Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    vis_net = pyvis.network.Network(
        height=height, directed=False, notebook=True, cdn_resources="in_line"
    )
    for node in surf.nodes:
        # if surf.amchi_mapping is None:
        vis_net.add_node(node.label, label=node.label, title=str(node.label))
    # else:
    #     chi = automol.amchi.join(list(map(surf.amchi_mapping.get, node.names_list)))
    #     image_path = _image_file_from_amchi(chi, out_dir=out_dir, stereo=stereo)
    #     vis_net.add_node(
    #         node.id,
    #         label=node.label,
    #         title=str(node.id),
    #         shape="image",
    #         image=image_path,
    #     )
    for edge in surf.edges:
        vis_net.add_edge(*edge.well_labels, title=edge.label)

    # Generate the HTML file
    vis_net.write_html(str(out_dir / out_name), open_browser=open_browser)


# Helpers
def _image_file_from_amchi(chi, out_dir: str | Path, stereo: bool = True):
    """Create an SVG molecule drawing and return the path."""
    out_dir = Path(out_dir)
    img_dir = Path("img")
    (out_dir / img_dir).mkdir(exist_ok=True)

    gra = automol.amchi.graph(chi, stereo=stereo)
    svg_str = automol.graph.svg_string(gra, image_size=100)

    chk = automol.amchi.amchi_key(chi)
    path = img_dir / f"{chk}.svg"
    with open(out_dir / path, mode="w") as file:
        file.write(svg_str)

    return str(path)
