import DeterminarOpinion as dt
import helpers
import os

def iniciar():

    tweets = dt.determina_opinion('datas\calentamientoClimatico.csv')
    tweets.preparacion()

    while True:

         # os.system('cls')
        print("========================")
        print(" BIENVENIDO AL Manager ")
        print("========================")
        print("[1] Aprendizaje multinomial ")
        print("[2] Aprendizaje SVM ")
        print("[3] Aprendizaje SVM Avanzado ")
        print("[4] Salir ")
        print("========================")

        opcion = input("> ")

        # helpers.limpiar_pantalla()


        if opcion == 1:
            while True:
                tweets.Aprendizaje_multinomial()
                helpers.limpiar_pantalla()
                print("========================")
                print("[1] Predecir frase")
                print("[2] No predecir frase")
                print("========================")

                opcion_1 = input("> ")
                # helpers.limpiar_pantalla()

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




