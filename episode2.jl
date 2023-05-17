
using DataFrames, Random

include("enviroment.jl")
include("model.jl")

df = DataFrame(CSV.File("./data/real_scenario.csv"))


# CSV.write("data.csv", df)
day = Int(24*(60/15))



for epsilon in [0.5]
    for decay_rate  in [0.25, 0.5, 0.7, 0.9,0.98]
        for learning_rate in [0.98, 0.9, 0.7, 0.5, 0.25]
            for forget_factor in [0.25, 0.5, 0.7, 0.9,0.98]
                for tuple_size in [10, 20, 40, 80, 160, 320, 720]
                    try
                        if ~isfile("./results/ESANN/julia/repisodes2/epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_forget-factor=$(forget_factor)_tuple-size=$(tuple_size)/checkpoint_2500.csv")

                            models = [generate_Model(1320, tuple_size, forget_factor) for i in 1:5]

                            encoders = get_encoders(df)
                            println("epsilon=$(epsilon)_learning-rate=$(learning_rate)_decay-rate=$(decay_rate)_forget-factor=$(forget_factor)_tuple-size=$(tuple_size)")
                        
                            for index in 1:2500
                                @time begin
                                    j = Int(rand(1:364))
                                    i = (j-1)*day + 1
                                
                                    run_episode(df[i:(i + day),:], "./results/ESANN/julia/repisodes2/", models, encoders, tuple_size, forget_factor, epsilon, learning_rate, decay_rate, index)
                                end
                            end
                        end
                    catch y
                        warn("Exception: ", y) # What to do on error.
                    end
                end
            end
        end
    end
end
