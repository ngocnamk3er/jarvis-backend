"""Flows a user builds from nodes (input, LLM, tool, output) and runs, compiled to a LangGraph graph.

schema.py    the flow format and what makes one valid
compiler.py  a valid flow -> a compiled LangGraph StateGraph
runner.py    running one: its result, or its events as they happen
catalog.py   the tools a Tool node may call
"""
