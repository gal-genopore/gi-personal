import pytest
import os
import xml.etree.ElementTree as ET
from pneumatic_symbols_LibreDraw import SvgRenderer, OdfRenderer

def test_svg_renderer_bounds():
    """Test that the SVG renderer accurately updates bounding boxes."""
    renderer = SvgRenderer()
    renderer.draw_rect(10, 20, 50, 60)
    
    min_x, min_y, max_x, max_y = renderer.get_bounds()
    assert min_x == 10
    assert min_y == 20
    assert max_x == 50
    assert max_y == 60

def test_svg_rendering_output():
    """Verify generated SVG matches expected basic structural elements."""
    renderer = SvgRenderer()
    renderer.draw_circle(100, 100, 15)
    svg_string = renderer.get_svg(view_box=(0, 0, 200, 200))
    
    assert '<circle cx="100" cy="100" r="15"' in svg_string
    assert 'viewBox="0 0 200 200"' in svg_string

def test_odf_renderer_fragment():
    """Test that OdfRenderer generates valid XML with correctly placed elements."""
    px_to_cm = 2.54 / 96.0
    renderer = OdfRenderer(0, 0, 100, 100, px_to_cm)
    
    renderer.draw_line(0, 0, 100, 100)
    glue_points = [(50, 50)]
    xml_fragment = renderer.get_xml_fragment(glue_points)
    
    # 1. Structural assertions remain the same
    assert 'svg:x="0.0000cm"' in xml_fragment
    assert 'svg:y="0.0000cm"' in xml_fragment
    assert 'draw:glue-point' in xml_fragment

    # 2. Add namespace declarations to the root tag for validation
    namespaces = (
        'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
        'xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"'
    )
    
    try:
        # Wrap fragment with valid namespaces so ElementTree can parse it
        ET.fromstring(f"<root {namespaces}>{xml_fragment}</root>")
    except ET.ParseError as e:
        import pytest
        pytest.fail(f"OdfRenderer generated malformed XML syntax: {e}")