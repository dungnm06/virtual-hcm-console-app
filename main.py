from dialog_manager.virtual_hcm import VirtualHCMChatbot

if __name__ == '__main__':
    chatbot = VirtualHCMChatbot()
    print("Start talking with the bot (type quit to stop)!")
    while True:
        input_sentence = input("You: ")
        if input_sentence.lower() == "quit":
            break

        response = chatbot.chat(input_sentence)
        print(response)
