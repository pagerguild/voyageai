package voyageai

type embeddingRequest struct {
	Input []string `json:"input"`
	Model string `json:"model"`
	InputType *string `json:"input_type,omitempty"`
	Truncation *bool `json:"truncation,omitempty"`
	OutputDimension *int `json:"output_dimension,omitempty"`
	OutputDType *string `json:"output_dtype,omitempty"`
	EncodingFormat *string `json:"encoding_format,omitempty"`
}

type EmbeddingRequestOpts struct {
	InputType *string `json:"input_type,omitempty"`
	Truncation *bool `json:"truncation,omitempty"`
	OutputDimension *int `json:"output_dimension,omitempty"`
	OutputDType *string `json:"output_dtype,omitempty"`
	EncodingFormat *string `json:"encoding_format,omitempty"`
}

type EmbeddingObject struct {
	Object string `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index int `json:"index"`
}

type UsageObject struct {
	TotalTokens int `json:"total_tokens"`
	ImagePixels *int `json:"image_pixels,omitempty"`
	TextTokens *int `json:"text_tokens,omitempty"`
}

type EmbeddingResponse struct {
	Object string `json:"object"`
	Data []EmbeddingObject `json:"data"`
	Model string `json:"model"`
	Useage UsageObject `json:"useage"`
}

type multimodalRequest struct {
	Inputs []string `json:"inputs"`
	Model string `json:"model"`
	InputType *string `json:"input_type,omitempty"`
	Truncation *bool `json:"truncation,omitempty"`
	OuputEncoding *string `json:"output_encoding,omitempty"`
}

type MultimodalRequestOpts struct {
	InputType *string `json:"input_type,omitempty"`
	Truncation *bool `json:"truncation,omitempty"`
	OuputEncoding *string `json:"output_encoding,omitempty"`
}

type APIError struct {
	Detail string `json:"detail"`
}

type rerankRequest struct {
	Query string `json:"query"`
	Documents []string `json:"documents"`
	Model string `json:"model"`
	TopK *int `json:"top_k,omitempty"`
	ReturnDocuments *bool `json:"return_documents,omitempty"`
	Truncation *bool `json:"truncation,omitempty"`
}

type RerankOpts struct {
	TopK *int `json:"top_k,omitempty"`
	ReturnDocuments *bool `json:"return_documents,omitempty"`
	Truncation *bool `json:"truncation,omitempty"`
}

type RerankObject struct {
	Index int `json:"index"`
	RelevanceScore float32 `json:"relevance_score"`
	Document *string `json:"document,omitempty"`
}

type RerankResponse struct {
	Object string `json:"object"`
	Data []RerankObject `json:"data"`
	Model string `json:"model"`
	Useage UsageObject `json:"useage"`
}
