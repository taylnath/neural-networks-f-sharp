open System
open Numpy

type Network =
    {
        numLayers: int
        sizes: int list // num neurons in each layer
        biases: NDarray list
        weights: NDarray list
    }
    
let rec zip (lis1: 'a list) (lis2: 'b list) =
    match lis1, lis2 with
    | [], [] -> []
    | [], [_] -> []
    | [_], [] -> []
    | [x], [y] -> [x,y]
    | x::xs, y::ys -> [x,y] @ zip xs ys
    | _ -> []
    
let makeNetwork (sizes: int list) =
    {
        numLayers = sizes.Length
        sizes = sizes
        biases = [for y in sizes.[1..] do np.random.randn(y, 1)]
        weights = [for x,y in zip sizes.[..(sizes.Length - 1)] sizes.[1..] do np.random.randn(y, x)]
    }
    
let sigmoid z = 1.0 / (1.0 + np.exp (-z))

/// Return output of network given input
let feedforward (network:Network) (input:NDarray) =
    let biasWeightPairs = zip network.biases network.weights
    // could use fold2 here instead of zip
    List.fold (fun output (bias, weight) -> sigmoid(np.dot(weight,output) + bias)) input biasWeightPairs
    
/// Return number of test inputs for which the neural network outputs the correct result
let evaluate network testData =
    let testResults = [for (x, y) in testData do np.argmax(feedforward network x), y]
    List.fold (fun acc (x,y) -> acc + if x = y then 1 else 0) 0 testResults
    
/// 
/// Return vector of partial derivatives partial C_x / partial a for the output activations
let costDerivative network outputActivations y = outputActivations - y

/// derivative of sigmoid
let sigmoidPrime z = (sigmoid z) * (1 - sigmoid z)
    
/// Return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x.
let backprop (network:Network) (x:NDarray) (y:NDarray) =
    // feedforward
    let activation = x
    let zs, activations =
        network
        |> (fun net -> List.zip network.biases network.weights)
        |> List.mapFold (fun activation (bias, weight) ->
            let z = np.dot(weight, activation) + bias
            (z, sigmoid z), sigmoid z ) x
        |> fst
        |> List.unzip
    // backward pass
    let delta = (costDerivative network activations.[activations.Length - 1] y) * (sigmoidPrime zs.[zs.Length - 1])
    let nabla_b, nabla_w = ([delta], [np.dot(delta, activations.[activations.Length - 2].transpose())])
    
    [2..network.numLayers - 1]
    |> List.fold (fun (nabla_b, nabla_w) l ->
        let z = zs.[zs.Length - l]
        let sp = sigmoidPrime z
        let delta = np.dot(network.weights.[network.weights.Length - l + 1].transpose(), delta) * sp
        (delta::nabla_b, np.dot(delta, activations.[activations.Length - l - 1].transpose())::nabla_w)
        ) (nabla_b, nabla_w)
        
/// Update the network's weights and biases by applying gradient descent using backpropagation to a single batch.
/// The miniBatch is a list of tuples (x,y) and eta is the learning rate
let updateMiniBatch network (miniBatch: (NDarray * NDarray) list) (eta:int) =
    let nabla_b0 = [for b in network.biases do np.zeros b.shape]
    let nabla_w0 = [for w in network.weights do np.zeros w.shape]
    let (nabla_b:NDarray list), (nabla_w:NDarray list) =
        miniBatch
        |> List.fold (fun (nabla_b, nabla_w) (x, y) ->
            let delta_nabla_b, delta_nabla_w = backprop network x y
            (
                [for nb, dnb in (List.zip nabla_b delta_nabla_b) do nb + dnb],
                [for nw, dnw in (List.zip nabla_w delta_nabla_w) do nw + dnw]
                )) (nabla_b0, nabla_w0)
    {
        network with
            weights = [for w, nw in (List.zip network.weights nabla_w) do w - ((eta / miniBatch.Length) * nw)]
            biases = [for b, nb in (List.zip network.biases nabla_b) do b - ((eta / miniBatch.Length) * nb)]
    }
    
let optionLen (oList:'a seq option) =
    match oList with
    | Some lis -> Seq.length lis
    | None -> 0
    
let extractTestData (someTestData: 'a seq option) =
    match someTestData with
    | Some testData -> testData
    | None -> Seq.empty<'a>
    
let SGD (network:Network) (trainingData:(NDarray * NDarray) list) (epochs:int) (miniBatchSize:int) (eta:int) (testData:('a * NDarray) seq option) =
    let n = trainingData.Length
    let nTest = optionLen testData
    let rnd = System.Random ()
    
    [0..epochs - 1]
    |> List.fold (fun net epoch ->
        trainingData
        |> List.sortBy (fun _ -> rnd.Next(0, epochs - 1))
        |> (fun arr -> [
            for k in 0 .. miniBatchSize .. n do arr.[k..k+miniBatchSize]
        ])
        |> List.fold (fun net miniBatch -> updateMiniBatch net miniBatch eta) net
        |> (fun net ->
            if testData <> None
            then printfn $"Epoch {epoch} : {evaluate net (extractTestData testData)} / {nTest}"
            else printfn $"Epoch {epoch} complete"
            net
            )
        ) network
    
let from whom =
    sprintf "from %s" whom

[<EntryPoint>]
let main argv =
    let message = from "F#" // Call the function
    printfn "Hello world %s" message
    0 // return an integer exit code