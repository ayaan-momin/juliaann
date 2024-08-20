using Flux, Images,  MLDatasets, Plots
using Flux: crossentropy, onecold, onehotbatch, train!, params
using LinearAlgebra, Random, Statistics

x_train_raw, y_train_raw = MLDatasets.MNIST(:train)[:]
x_test_raw, y_test_raw = MLDatasets.MNIST(:test)[:]


x_train = Flux.flatten(x_train_raw)
x_test = Flux.flatten(x_test_raw)

y_train = onehotbatch(y_train_raw,0:9)
y_test = onehotbatch(y_test_raw,0:9)


model = Chain(
    Dense(28*28,32,relu),
    Dense(32,10),
    softmax
)

loss(x,y) = crossentropy(model(x),y)
ps = params(model)
learning_rate = 0.01
opt = ADAM(learning_rate)

#training 

loss_history = []
epochs = 500

for epoch in 1:epochs
    train!(loss,ps,[(x_train,y_train)],opt)
    train_loss = loss(x_train,y_train)
    push!(loss_history,train_loss)
    println("Epoch = $epoch : Training Loss = $train_loss")
end

#prdictions

y_hat_raw = model(x_test)
y_hat = onecold(y_hat_raw) .- 1
y = y_test_raw
mean(y_hat .== y)

check = [y_hat[i] == y[i] for i in eachindex(y)]
index = collect(1:length(y))
check_display = [index y_hat y check]
vscodedisplay(check_display)

# initialize plot

gr(size = (600, 600))

# plot learning curve

p_l_curve = plot(1:epochs, loss_history,
    xlabel = "Epochs",
    ylabel = "Loss",
    title = "Learning Curve",
    legend = false,
    color = :blue,
    linewidth = 2
)
savefig(p_l_curve, "learning_curve.svg")