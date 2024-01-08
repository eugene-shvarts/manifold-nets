### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 6cf50f8a-ab5b-11ee-3ab1-31e32a50824a
begin
	using Pkg
	Pkg.activate(".")
end

# ╔═╡ c7e21c47-92da-4481-916f-12580a5d0a93
using HDF5, Chain, DataFrames, DataStructures, Plots, LinearAlgebra, StatsBase, SparseArrays, Unzip, Distances

# ╔═╡ c80f6710-fbbc-4af7-b3d2-2f29f7826b5f
tf = h5read("../../data/text_features.h5", "text_features")

# ╔═╡ d288688b-23fc-4342-97d8-cd0a45f43c7a
tfcolnorm = tf ./ norm.(eachcol(tf))'

# ╔═╡ 580f8d59-699d-490e-8ce0-bb05fadd4c5c
[
	collect(partialsortperm(abs.(col), 1:10; rev=true))
	for col in eachcol(tfcolnorm)
]

# ╔═╡ 6adfce24-50a1-4002-9daa-79d260ea4467
falloffs = sort.(eachcol(abs.(tfcolnorm)); rev=true)

# ╔═╡ 472bd81d-1be1-4549-9b4b-c5d48fbe328b
sum(
	v[6] > 1.5e-1
	for v in falloffs
)

# ╔═╡ dc0e1ce5-796b-44d3-b1b2-91d365f55a0d
plot(hcat([
	falloffs[i]
	for i in rand(1:length(falloffs), 100)
]...), legend=false)

# ╔═╡ 7054a66f-6130-4049-992a-665acf6aec3b
falloffs[:,rand(1:size(falloffs, 2), 100)]

# ╔═╡ d58b9dc2-f7be-4d29-bd15-72b32355ddab
rand(1:size(falloffs, 2), 100)

# ╔═╡ 196c1f14-3fd4-401a-8ebc-78e8fd9949b4
size(falloffs)

# ╔═╡ 35816d72-9953-4165-898c-7c66e25322f2
plot(tf[:,1])

# ╔═╡ 1ba0de01-c226-4c01-959b-ae69e320b22b
tops = [
	collect(partialsortperm(col, 1:5; rev=true))
	for col in eachcol(abs.(tfcolnorm))
]

# ╔═╡ c72adfb5-5605-4de1-b3a7-3a196ab940e9
size(tf)

# ╔═╡ cec2c549-c2e2-42e9-a7af-a0908cd0a14f
findall(v -> 313 ∉ v,tops)

# ╔═╡ 09b11cb4-556b-4e2d-9ee6-0d12afe28e1e
words = readlines("../../data/text_words.txt")

# ╔═╡ 762857c5-2b05-422a-8b20-21458171943b
words[77765]

# ╔═╡ b5fcf300-0884-4bd8-9740-c406346baca3
partialsortperm(abs.(tfcolnorm[:,77765]), 1:10; rev=true)

# ╔═╡ cef7ab40-4d57-4616-a49c-ec93cad0725e
77765 |> n -> begin
	plot(
		tfcolnorm[:,n],
		legend=false,
		title=words[n]
	)
	vline!([313, 134, 93, 330, 8])
end

# ╔═╡ 7876cd6f-b898-4140-90ac-934fabd13f03
sort(collect(countmap(x for v in tops for x in v)), by=x -> -x[2])

# ╔═╡ 6598bd30-fd68-456c-8177-77c587854f91
plot(sort(collect(values(countmap(x for v in tops for x in v))), rev=true), yscale=:log10)

# ╔═╡ 16bb9506-365a-40d0-baf6-fac4a24af1d2
function thresh_distmat(X, σ = 1, k = 1)
	t = σ*k
	entries = Tuple{Int, Int, Float64}[]

	for (i,u) in enumerate(eachcol(X))
		for (j,v) in enumerate(eachcol(X[:,(i+1):end]))
			d = sum((u-v).^2)
			if d < t
				push!(entries, (i,i+j, exp(-d/σ)))
			end
		end
	end

	sparse(unzip(entries)...)
end

# ╔═╡ 7d963a26-91a3-49ac-8602-10830d6a4a4c
# W = thresh_distmat(tfcolnorm[:,1:20_000], 0.2, 0.5)

# ╔═╡ 062db28f-05e5-4b3b-bfbc-4a7dce8df15b
pairwise(SqEuclidean(), tfcolnorm[:,1:20_000], dims=2)

# ╔═╡ Cell order:
# ╠═6cf50f8a-ab5b-11ee-3ab1-31e32a50824a
# ╠═c7e21c47-92da-4481-916f-12580a5d0a93
# ╠═c80f6710-fbbc-4af7-b3d2-2f29f7826b5f
# ╠═d288688b-23fc-4342-97d8-cd0a45f43c7a
# ╠═580f8d59-699d-490e-8ce0-bb05fadd4c5c
# ╠═6adfce24-50a1-4002-9daa-79d260ea4467
# ╠═472bd81d-1be1-4549-9b4b-c5d48fbe328b
# ╠═dc0e1ce5-796b-44d3-b1b2-91d365f55a0d
# ╠═7054a66f-6130-4049-992a-665acf6aec3b
# ╠═d58b9dc2-f7be-4d29-bd15-72b32355ddab
# ╠═196c1f14-3fd4-401a-8ebc-78e8fd9949b4
# ╠═35816d72-9953-4165-898c-7c66e25322f2
# ╠═1ba0de01-c226-4c01-959b-ae69e320b22b
# ╠═c72adfb5-5605-4de1-b3a7-3a196ab940e9
# ╠═cec2c549-c2e2-42e9-a7af-a0908cd0a14f
# ╠═09b11cb4-556b-4e2d-9ee6-0d12afe28e1e
# ╠═762857c5-2b05-422a-8b20-21458171943b
# ╠═b5fcf300-0884-4bd8-9740-c406346baca3
# ╠═cef7ab40-4d57-4616-a49c-ec93cad0725e
# ╠═7876cd6f-b898-4140-90ac-934fabd13f03
# ╠═6598bd30-fd68-456c-8177-77c587854f91
# ╠═16bb9506-365a-40d0-baf6-fac4a24af1d2
# ╠═7d963a26-91a3-49ac-8602-10830d6a4a4c
# ╠═062db28f-05e5-4b3b-bfbc-4a7dce8df15b
