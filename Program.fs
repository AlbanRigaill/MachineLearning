// Learn more about F# at http://fsharp.org

open System
open Microsoft.ML.Data
open Microsoft.ML
open Microsoft.ML.Transforms

[<CLIMutable>]
type Digit = {
    [<LoadColumn(0)>] Number: float32
    [<LoadColumn(1,784)>] [<VectorType(784)>] PixelValues: float32[]
}

[<CLIMutable>]
type DigitPrediction = {
    Score: float32[]
}

let trainDataPath = sprintf "%s\\mnist_train.csv" Environment.CurrentDirectory
let testDataPath = sprintf "%s\\mnist_test.csv" Environment.CurrentDirectory


[<EntryPoint>]
let main argv =
    
    let context = new MLContext();
    let trainData = context.Data.LoadFromTextFile<Digit>(trainDataPath, hasHeader = true, separatorChar = ',')
    let testData = context.Data.LoadFromTextFile<Digit>(testDataPath, hasHeader = true, separatorChar = ',')


    let pipeline =
        EstimatorChain()
            .Append(context.Transforms.Conversion.MapValueToKey("Label", "Number", keyOrdinality = ValueToKeyMappingEstimator.KeyOrdinality.ByValue))
            .Append(context.Transforms.Concatenate("Features", "PixelValues"))
            .AppendCacheCheckpoint(context)
            .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy())
            .Append(context.Transforms.Conversion.MapKeyToValue("Number", "Label"))


    let model = trainData |> pipeline.Fit
    let engine = context.Model.CreatePredictionEngine model

    let digits = context.Data.CreateEnumerable(testData, false) |> Array.ofSeq
    let testDigits = [ digits.[8]; digits.[13]; digits.[19]; digits.[24]; digits.[87]; digits.[123]; digits.[156]]

    printfn "Résultats"
    printf "  #\t\t"
    [0..9] |> Seq.iter(fun i -> printf "%i\t\t" i)
    printfn ""

    testDigits |> Seq.iter(
        fun digit ->
            printf "  %i\t\t" (int digit.Number)
            let r = engine.Predict digit
            r.Score |> Seq.iter(fun s -> printf "%f\t" s)
            printfn ""
    )


    0 // return an integer exit code
