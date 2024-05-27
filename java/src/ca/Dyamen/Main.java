package ca.Dyamen;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import javax.imageio.ImageIO;

public class Main
{
	private static final int GREY_SCALE = 0;
	private static final int POTENTIAL = 1;

	private static final int getIntFromPointer(final byte[] data, final int pointer)
	{
		int value = 0;
		int mask = 0xFF;
		for(int i = 0; i < 4; ++i, mask<<=8) value |= (data[pointer + i] << (i*8)) & mask;
		return value;
	}

	private static final int getByteFromPointer(final byte[] data, final int pointer)
	{
		int value = data[pointer];
		return value;
	}

	private static final BufferedImage draw_GREY_SCALE(int res, int max_itr, int max_itr_value, int[] scalar_field)
	{
		int[] image_data = new int[res*res];
		for(int i = 0; i < res*res; ++i)
		{
			int scale = (int) (255. * ((double) scalar_field[i]) / ((double) max_itr));
			byte gs = (byte) (scale & 0xFF);
			image_data[i] = ((gs << 16) & 0xFF0000) | ((gs << 8) & 0xFF00) | ((gs << 0) & 0xFF);
		}

		BufferedImage image = new BufferedImage(res, res, BufferedImage.TYPE_INT_RGB);
		image.setRGB(0, 0, res, res, image_data, 0, res);

		return image;
	}

	private static final BufferedImage draw_POTENTIAL(int res, int max_itr, int max_itr_value, int[] scalar_field)
	{
		final double ca = 3.321928095, cb = 0.782985961, cc = 0.413668099;
		int[] image_data = new int[res*res];
		for(int i = 0; i < res*res; ++i)
		{
			double x = Math.PI * ((double) scalar_field[i]) / ((double) max_itr);
			
			int r = (int) (255 * (1 - Math.cos(ca*x)) / 2);
			int g = (int) (255 * (1 - Math.cos(cb*x)) / 2);
			int b = (int) (255 * (1 - Math.cos(cc*x)) / 2);
			int color = ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF);
			image_data[i] = color;
		}
		
		BufferedImage image = new BufferedImage(res, res, BufferedImage.TYPE_INT_RGB);
		image.setRGB(0, 0, res, res, image_data, 0, res);

		return image;
	}
	
	private static final BufferedImage draw(int draw_method, int res, int max_itr, int max_itr_value, int[] scalar_field)
	{
		switch(draw_method)
		{
			case GREY_SCALE:	return draw_GREY_SCALE(res, max_itr, max_itr_value, scalar_field);
			case POTENTIAL:		return draw_POTENTIAL(res, max_itr, max_itr_value, scalar_field);
			default: 		return null;
		}
	}

	private static final void writeImage(BufferedImage image) throws IOException
	{
		ImageIO.write(image, "PNG", new File("output.png"));
	}

	public static void main(String... args) throws IOException
	{
		File fdata = new File("out.dat");
		final byte[] data = Files.readAllBytes(fdata.toPath());
		
		int mode = getByteFromPointer(data, 0);
		int max_itr = getIntFromPointer(data, 1);
		int res = getIntFromPointer(data, 5);
		System.out.printf("mode: %6x | max_itr: %6x | res: %6x\n", mode, max_itr, res);

		int[] scalar_field = new int[res*res];
		
		System.out.println("---" + data.length);

		int max_value = 0;
		for(int i = 0; i < data.length - 9; i+=4)
		{
			scalar_field[i/4] = getIntFromPointer(data, i + 9);
			max_value = Math.max(scalar_field[i/4], max_value);
		}

		BufferedImage image = draw(mode, res, max_itr, max_value, scalar_field);

		writeImage(image);
	}
}
