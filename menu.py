import DeterminarOpinion as dt
import helpers

def menu():
    print("========================")
    print(" BIENVENIDO AL Manager ")
    print("========================")
    print("[1] Aprendizaje multinomial ")
    print("[2] Aprendizaje SVM ")
    print("[3] Aprendizaje SVM Avanzado ")
    print("[4] Salir ")
    print("========================")

def iniciar():
    tweets = dt.determina_opinion('calentamientoClimatico.csv')
    tweets.preparacion()
    helpers.limpiar_pantalla()

    while True:

        menu()
        opcion = int(input("> "))
        helpers.limpiar_pantalla()

        if opcion == 1:
            tweets.Aprendizaje_multinomial()
            while True:
                print("========================")
                print("[1] Predecir frase")
                print("[2] Volver")
                print("========================")

                opcion_1 = int(input("> "))
                helpers.limpiar_pantalla()

                if opcion_1 == 1:
                    print('Introduce una frase(en inglés)')
                    frase = input("> ")
                    tweets.predecir_frase(frase)

                if opcion_1 == 2:
                    break

        if opcion == 2:
            tweets.aprendizaje_svm()

        if opcion == 3:
            tweets.aprendizaje_svm_avanzado()

        if opcion == 4:
            print("Saliendo...\n")
            break


