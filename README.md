# Voyage Go Library
`voyageai` is a Go client for [Voyage AI](https://www.voyageai.com/)
## Installation
```bash
go get github.com/pagerguild/voyageai
```
## Usage
### Generating Embeddings
```go
	vo := voyageai.NewClient(nil)
	
	embeddings, err := vo.Embed(
		[]string{
			"Embed this text please",
			"And this as well",
		}, 
		"voyage-3-lite", 
		nil,
	)

	if err != nil {
		fmt.Printf("Could not get embedding: %s", err.Error())
	}
	// ... Use the generated embeddings ...
```

If the embedding request is successful, the `embeddings` variable
will contain an `EmbeddingResponse`, which contains the embedding objects and usage details.

```go
type EmbeddingObject struct {
	Object string `json:"object"` // The object type, which is always "embedding".
	Embedding []float32 `json:"embedding"` // An array of embedding objects.
	Index int `json:"index"` // An integer representing the index of the embedding within the list of embeddings.
}

type UsageObject struct {
	TotalTokens int `json:"total_tokens"` // The total number of tokens used for computing the embeddings.
	ImagePixels *int `json:"image_pixels,omitempty"` // The total number of image pixels in the list of inputs.
	TextTokens *int `json:"text_tokens,omitempty"` // The total number of text tokens in the list of inputs.
}

type EmbeddingResponse struct {
	Object string `json:"object"` // The object type, which is always "list".
	Data []EmbeddingObject `json:"data"` // An array of embedding objects.
	Model string `json:"model"` // Name of the model.
	Useage UsageObject `json:"useage"` // An object containing useage details
}
```

### Generating Multimodal Embeddings
```go
	// png, jpeg, and gif are all supported file types, webp is not yet supported.
	img, err := os.Open("path/to/image.png")
	if err != nil {
		fmt.Printf("Could not open image: %s", err.Error())
	}

	multimodalInput := []voyageai.MultimodalContent{
		{
			Content: []voyageai.MultimodalInput{
				{
					Type: "text",
					Text: "This is a picture of the Go mascot",
				},
				{
					Type: "image_base64",
					ImageBase64: voyageai.MustGetBase64(img),
				},
			},
		},
	}

	mEmbedding, err := vo.MultimodalEmbed(multimodalInput, "voyage-multimodal-3", nil)
	if err != nil {
		fmt.Printf("Could not get multimodal embedding: %s", err.Error())
	}
	// ... Use the generated embeddings ...
``` 

A successful multimodal embedding request also returns an `EmbeddingResponse`.

### Reranking
```go
	vo := voyageai.NewClient(nil)

	reranking, err := vo.Rerank(
		"This is an example query",
		[]string{"this is a document", "this is also a document"}, 
		"rerank-2-lite", 
		nil,
	)
	if err != nil {
		fmt.Printf("Could not get reranking results: %s", err.Error())
	}
	// ... Use the reranked documents  ...
```

If the reranking request is successful, the `reranking` variable
will contain a `RerankingResponse`, which contains the reranking objects and usage details.

```go
type RerankResponse struct {
	Object string `json:"object"` // The object type, which is always "list".
	Data []RerankObject `json:"data"` // An array of the reranking results, sorted by the descending order of relevance scores.
	Model string `json:"model"` // Name of the model.
	Useage UsageObject `json:"useage"` // An object containing useage details
}


type RerankObject struct {
	Index int `json:"index"` // The index of the document in the input list.
	RelevanceScore float32 `json:"relevance_score"` // The relevance score of the document with respect to the query.
	Document *string `json:"document,omitempty"` // The document string. Only returned when return_documents is set to true.
}
```

