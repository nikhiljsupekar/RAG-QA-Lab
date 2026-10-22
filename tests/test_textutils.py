from rag.textutils import md_escape


def test_markdown_special_characters_are_escaped():
    assert md_escape("a*b*_c_ #d $5$ `x` [s · chunk-0001]") == \
        r"a\*b\*\_c\_ \#d \$5\$ \`x\` \[s · chunk-0001\]"


def test_numbered_heading_is_not_turned_into_a_list():
    assert md_escape("3. Remote Work") == r"3\. Remote Work"
    assert md_escape("version 1.7 ok") == "version 1.7 ok"  # mid-line numbers untouched


def test_newlines_become_hard_line_breaks():
    assert md_escape("one\ntwo") == "one  \ntwo"


def test_plain_text_is_unchanged():
    assert md_escape("Employees may work remotely up to 3 days per week.") == \
        "Employees may work remotely up to 3 days per week."
