# Random Ensemble Semisupervised Density Agglomerative Clustering (RESDAC)

# Semisupervised Illumination Density Agglomerative Clustering (SIDAC)

# Import, then bring into the global namespace

using Base.Iterators

using LinearAlgebra
using Statistics

using MLStyle
using Lazy

# For predicting with a RESDAC model
function RESDAC_predict(item, idsarray, data, labels)
  map(
    ids ->
      SIDAC_predict(item, ids, data, labels),
    idsarray
  )
end

# For training a RESDAC model
function RESDAC_train(data, labels, n_models)
  map(
    _ ->
      SIDAC_train(RESDAC_segment(data, labels), labels),
    1 : n_models
  )
end

# For gathering a segment of the data
function RESDAC_segment(data, labels)
  fdict_tuples = filter(
    item ->
      item[1] != nothing,
    zip(labels, data)
  )
  fdict_pairs = map(
    ( @λ (a, b) ->
      a => b
    ),
    fdict_tuples
  )
  data_sections = Dict(fdict_pairs...)
  opt_labels = unique(filter(
    item ->
      item != nothing,
    labels
  ))
  collect(flatten(map(
    label ->
      rand_select_ratio(1 / 3, data_sections[label]),
    opt_label
  )))
end

# For predicting with a SIDAC model
function SIDAC_predict(item, ids, data, labels)
  item_index = index_of(data, item)
  item_id = ids[item_index]
  map(
    index ->
      ids[index] == item_id ? labels[index] : nothing,
    1 : length(ids)
  )
end

# For training a SIDAC model
function SIDAC_train(data, labels)
  scan = filter(x -> !same(x), product([data, data]))
  dims = length(data[1])
  scores = map(
    ( @λ [a, b] ->
      SIDAC_calc_score(a, b, dims, data)
    ),
    scan
  )
  SIDAC_training_iter(scan, scores, collect(1 : length(data)), [], data, labels)
end

# Iteration of training a SIDAC model
@rec function SIDAC_training_iter(scan, scores, ids, already, data, labels)
  available_indices = filter(
    index ->
      !(index in already),
    1 : length(scores)
  )
  available_scores = just_these_indices(scores, available_indices)
  chosen_score = maximum(available_scores)
  chosen_score_index = index_of(scores, chosen_score)
  orig_indices = map(
    item ->
      index_of(data, item),
    scan[chosen_score_index]
  )
  chosen_and_target_indices = let
    exval(labels[orig_indices[2]]) ? orig_indices : reverse(orig_indices)
  end
  chosen_index = chosen_and_target_indices[1]
  target_index = chosen_and_target_indices[2]
  chosen_id = ids[target_index]
  f_labels_none = map(
    item ->
      labels[item] == nothing,
    merged(chosen_index, chosen_id, ids)
  )
  done = !(1 in f_labels_none)
  if done
    merged(chosen_index, chosen_id, ids)
  else
    SIDAC_training_iter(
      scan, scores,
      merged(chosen_index, chosen_id, ids),
      [[chosen_score_index]; already],
      data, labels
    )
  end
end

# Calculate the score of a potential merge for a SIDAC model
function SIDAC_calc_score(a, b, dims, data)
  if a == b
    0
  else
    t_dist = norm(a - b)
    raw_d_delta = SIDAC_density(b, dims, data) - SIDAC_density(a, dims, data)
    delta_density = raw_d_delta / t_dist
    delta_density / (t_dist ^ 3)
  end
end

# Calculate the illumination density for a SIDAC model
function SIDAC_density(point, dims, data)
  dists = map(
    item ->
      norm(item - point),
    data
  )
  recieved = map(
    dist_ ->
      (dist_ == 0.0) ? 0 : (1 / (dist_ ^ dims)),
    dists
  )
  sum(recieved)
end

# Are all the elements the same
function same(coll)
  all(map(
    item ->
      item == coll[1],
    coll
  ))
end

# Index of an element in a iter
function index_of(itr, desired_item)
  (findall(
    item ->
      item == desired_item,
    itr)
  )[1]
end

# Is this a value
function exval(value)
  value != nothing
end

# Cartesian product of two
function product(coll)
  map(
    collect,
    collect(Iterators.product(coll...))
  )
end

# FlatMap for Union{T, Nothing} optional type
function opt_nothing_map(f, optval)
  if optval == nothing
    nothing
  else
    f(optval)
  end
end

# Filter by indices
function just_these_indices(coll, indices)
  filter(
    item ->
      index_of(coll, item) in indices,
    coll
  )
end

# Merge IDs in a collection
function merged(index, value, coll)
  indexed_value = coll[index]
  replace(coll, indexed_value => value)
end

# Randomly select a ratio of an array
function rand_select_ratio(r, coll)
  len_c = length(coll)
  just_these_indices(
    coll,
    map(
      _ ->
        rand(1 : len_c),
      1 : Int(floor(r * len_c))
    )
  )
end

