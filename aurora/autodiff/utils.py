import numpy as np
from collections import deque
from aurora.autodiff.autodiff import PlaceholderOp


def find_topo_sort(node_list):
    """

    :param node_list:
    :return:
    """
    visited = set()
    topo_order = []
    for node in node_list:
        depth_first_search(node, visited, topo_order)
    return topo_order


def depth_first_search(node, visited, topo_order):
    """

    :param node:
    :param visited:
    :param topo_order:
    :return:
    """
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        depth_first_search(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """
    Custom sum function in order to avoid
    create redundant nodes in Python sum implementation
    :param node_list:
    :return:
    """
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


def to_d2lang(end_nodes, start_nodes=None, graph_name="AuroraGraph"):
    # 1. 反向收集所有祖先节点
    all_ancestors = set()
    queue = deque(end_nodes)
    while queue:
        node = queue.popleft()
        if node in all_ancestors:
            continue
        all_ancestors.add(node)
        for inp in node.inputs:
            if inp not in all_ancestors:
                queue.append(inp)

    if start_nodes is None:
        start_nodes = [n for n in all_ancestors if not n.inputs]
    else:
        start_nodes = [n for n in start_nodes if n in all_ancestors]

    # 2. 构建反向邻接表（用于前向可达）
    rev_adj = {n: [] for n in all_ancestors}
    for node in all_ancestors:
        for inp in node.inputs:
            if inp in all_ancestors:
                rev_adj[inp].append(node)

    # 3. 从起始节点前向可达子图
    forward_reach = set()
    queue = deque(start_nodes)
    while queue:
        node = queue.popleft()
        if node in forward_reach:
            continue
        forward_reach.add(node)
        for out_node in rev_adj.get(node, []):
            if out_node not in forward_reach:
                queue.append(out_node)

    sub_nodes = all_ancestors.intersection(forward_reach)
    for n in end_nodes:
        sub_nodes.add(n)

    # 分配自增 ID（按内存 id 排序保证稳定）
    sorted_nodes = sorted(sub_nodes, key=lambda n: id(n))
    node_to_id = {node: i for i, node in enumerate(sorted_nodes)}

    # 4. 生成 D2 代码
    # 先定义 classes（根级别，在图外部）
    lines = [
        "classes: {",
        "  input: {",
        "    shape: rectangle",
        "    style: {fill: deepskyblue; stroke: steelblue; stroke-width: 2; border-radius: 4; font-color: white; bold: true}",
        "  }",
        "  param: {",
        "    shape: rectangle",
        "    style: {fill: orange; stroke: darkorange; stroke-width: 2; border-radius: 4; font-color: white; bold: true}",
        "  }",
        "  op: {",
        "    shape: rectangle",
        "    style: {fill: lightgray; stroke: gray; stroke-width: 1; border-radius: 4; shadow: true}",
        "  }",
        "  loss: {",
        "    shape: hexagon",
        "    style: {fill: red; stroke: darkred; stroke-width: 2; border-radius: 4; font-color: white; bold: true; shadow: true}",
        "  }",
        "  dataflow: {",
        "    style: {stroke: dodgerblue; stroke-width: 1}",
        "  }",
        "}",
        "",
        f"{graph_name}: {{",
        '  style: {fill: "#fafafa"; stroke: "#dee2e6"; stroke-width: 1}',
    ]

    # 节点定义
    for node in sorted_nodes:
        node_id = f"n{node_to_id[node]}"
        op = node.op

        if isinstance(op, PlaceholderOp):
            if node.const is not None:
                label = f"Parameter\\n{node.name}\\n{node.const.shape}"
                css_class = "param"
            else:
                label = f"Input\\n{node.name}"
                css_class = "input"
            label = label.replace('"', '\\"')
            lines.append(f'  {node_id}: {{ class: {css_class}; label: "{label}" }}')
        else:
            class_name = op.__class__.__name__
            if "loss" in node.name.lower() or "CrossEntropy" in class_name:
                label = "Loss"
                css_class = "loss"
            else:
                label = class_name
                css_class = "op"
            label = label.replace('"', '\\"')
            lines.append(f'  {node_id}: {{ class: {css_class}; label: "{label}" }}')

    # 边定义（不带标签，使用 dataflow 类）
    for node in sorted_nodes:
        if node in start_nodes:
            continue
        dst_id = f"n{node_to_id[node]}"
        for inp in node.inputs:
            if inp in sub_nodes:
                src_id = f"n{node_to_id[inp]}"
                if src_id != dst_id:
                    lines.append(f"  {src_id} -> {dst_id}: {{ class: dataflow }}")

    lines.append("}")
    return "\n".join(lines)
