using ManifoldFlows
using Documenter

DocMeta.setdocmeta!(ManifoldFlows, :DocTestSetup, :(using ManifoldFlows); recursive=true)

makedocs(;
    modules=[ManifoldFlows],
    authors="Ben Murrell <murrellb@gmail.com> and contributors",
    repo="https://github.com/MurrellGroup/ManifoldFlows.jl/blob/{commit}{path}#{line}",
    sitename="ManifoldFlows.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MurrellGroup.github.io/ManifoldFlows.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/MurrellGroup/ManifoldFlows.jl",
    devbranch="main",
)
