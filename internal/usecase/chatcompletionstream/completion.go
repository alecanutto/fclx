package chatcompletionstream

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/alecanutto/fclx/chat-service/internal/domain/entity"
	"github.com/alecanutto/fclx/chat-service/internal/domain/gateway"
	openai "github.com/sashabaranov/go-openai"
)

type ChatCompletionConfigInputDTO struct {
	Model                string
	ModelMaxToken        int
	Temperature          float32
	TopP                 float32
	N                    int
	Stop                 []string
	MaxTokens            int
	PresencePenalty      float32
	FrequencyPenalty     float32
	InitialSystemMessage string
}

type ChatCompletionInputDTO struct {
	ChatID      string
	UserID      string
	UserMessage string
	Config      ChatCompletionConfigInputDTO
}

type ChatCompletionOutputDTO struct {
	ChatID  string
	UserID  string
	Content string
}

type ChatCompletionUseCase struct {
	ChatGateway  gateway.ChatGateway
	OpenAIClient *openai.Client
	Stream       chan ChatCompletionOutputDTO
}

func NewChatCompletionUseCase(chatGateway gateway.ChatGateway, openAIClient *openai.Client, stream chan ChatCompletionOutputDTO) *ChatCompletionUseCase {
	return &ChatCompletionUseCase{
		ChatGateway:  chatGateway,
		OpenAIClient: openAIClient,
		Stream:       stream,
	}
}

func (uc *ChatCompletionUseCase) Execute(ctx context.Context, input ChatCompletionInputDTO) (*ChatCompletionOutputDTO, error) {
	chat, err := uc.ChatGateway.FindChatByID(ctx, input.ChatID)
	if err != nil {
		if err.Error() == "chat not found" {
			chat, err = createNewChat(input)
			if err != nil {
				return nil, fmt.Errorf("error creating new chat: %s", err.Error())
			}
			err = uc.ChatGateway.CreateChat(ctx, chat)
			if err != nil {
				return nil, fmt.Errorf("error persisting new chat: %s", err.Error())
			}
		} else {
			return nil, fmt.Errorf("error fetching existing new chat: %s", err.Error())
		}
	}
	userMessage, err := entity.NewMessage("user", input.UserMessage, chat.Config.Model)
	if err != nil {
		return nil, fmt.Errorf("error creating user message: %s", err.Error())
	}
	err = chat.AddMessage(userMessage)
	if err != nil {
		return nil, fmt.Errorf("error adding new message: %s", err.Error())
	}
	messages := []openai.ChatCompletionMessage{}
	for _, msg := range chat.Messages {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}
	resp, err := uc.OpenAIClient.CreateChatCompletionStream(ctx, openai.ChatCompletionRequest{
		Model:            chat.Config.Model.Name,
		Messages:         messages,
		Temperature:      chat.Config.Temperature,
		TopP:             chat.Config.TopP,
		N:                chat.Config.N,
		Stop:             chat.Config.Stop,
		MaxTokens:        chat.Config.MaxTokens,
		PresencePenalty:  chat.Config.PresencePenalty,
		FrequencyPenalty: chat.Config.FrequencyPenalty,
		Stream:           true,
	})
	if err != nil {
		return nil, fmt.Errorf("error creating chat completion: %s", err.Error())
	}
	var fullResponse strings.Builder
	for {
		response, err := resp.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("error streaming response: %s", err.Error())
		}
		fullResponse.WriteString(response.Choices[0].Delta.Content)
		r := ChatCompletionOutputDTO{
			ChatID:  chat.ID,
			UserID:  input.UserID,
			Content: fullResponse.String(),
		}
		uc.Stream <- r
	}
	assistent, err := entity.NewMessage("assistent", fullResponse.String(), chat.Config.Model)
	if err != nil {
		return nil, fmt.Errorf("error creating assistent message: %s", err.Error())
	}
	err = chat.AddMessage(assistent)
	if err != nil {
		return nil, fmt.Errorf("error adding new message: %s", err.Error())
	}
	err = uc.ChatGateway.SaveChat(ctx, chat)
	if err != nil {
		return nil, fmt.Errorf("error saving chat: %s", err.Error())
	}
	return &ChatCompletionOutputDTO{
		ChatID:  chat.ID,
		UserID:  input.UserID,
		Content: fullResponse.String(),
	}, nil
}

func createNewChat(input ChatCompletionInputDTO) (*entity.Chat, error) {
	model := entity.NewModel(input.Config.Model, input.Config.ModelMaxToken)
	chatConfig := &entity.ChatConfig{
		Temperature:      input.Config.Temperature,
		TopP:             input.Config.TopP,
		N:                input.Config.N,
		Stop:             input.Config.Stop,
		MaxTokens:        input.Config.MaxTokens,
		PresencePenalty:  input.Config.PresencePenalty,
		FrequencyPenalty: input.Config.FrequencyPenalty,
		Model:            model,
	}
	initialMessage, err := entity.NewMessage("system", input.Config.InitialSystemMessage, model)
	if err != nil {
		return nil, fmt.Errorf("error creating initial message: %s", err.Error())
	}
	chat, err := entity.NewChat(input.UserID, initialMessage, chatConfig)
	if err != nil {
		return nil, fmt.Errorf("error creating new chat: %s", err.Error())
	}
	return chat, nil
}
