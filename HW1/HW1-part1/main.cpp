#include "mbed.h"

#define WAIT_TIME_MS 100

AnalogIn intTemp(ADC_TEMP);

int main() {
  while (true) {
    int value = intTemp.read_u16();
    float Vsense = (value * 3.3f) / 65535.0f;
    float temperature = ((Vsense - 0.76f) / 0.0025f) + 25.0f;
    printf("Temperature is %.2f C' \n", temperature);

    thread_sleep_for(WAIT_TIME_MS);
  }
}
