package main

import (
	"fmt"
	"os"

	"github.com/austinfhunter/voyageai"
)

func main() {
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

	fmt.Printf("%v\n", embeddings.Data[0].Embedding[0:5])

	img, err := os.Open("./assets/gopher.png")
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
					Type:        "image_base64",
					ImageBase64: voyageai.MustGetBase64(img),
				},
			},
		},
	}

	mEmbedding, err := vo.MultimodalEmbed(multimodalInput, "voyage-multimodal-3", nil)
	if err != nil {
		fmt.Printf("Could not get multimodal embedding: %s", err.Error())
	}
	fmt.Printf("%v\n", mEmbedding.Data[0].Embedding[0:5])

	reranking, err := vo.Rerank(
		"This is an example query",
		[]string{"this is a document", "this is also a document"},
		"rerank-2-lite",
		nil,
	)
	if err != nil {
		fmt.Printf("Could not get reranking results: %s", err.Error())
	}
	fmt.Printf("%v\n", reranking)
}
